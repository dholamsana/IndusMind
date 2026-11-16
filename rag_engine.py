
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# --------------------------------------------------
# 1. INITIALISE EMBEDDING MODEL AND CHROMADB
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path="./maintenance_db"
)

# Rebuild the collection with updated project knowledge
try:
    client.delete_collection("maintenance_knowledge")
except Exception:
    pass

collection = client.create_collection(
    name="maintenance_knowledge"
)

# --------------------------------------------------
# 2. PROJECT KNOWLEDGE BASE
# --------------------------------------------------

documents = [

    # Dataset information
    """
    IndusMind uses the NASA C-MAPSS FD001 Turbofan Engine
    Degradation Simulation Dataset.

    The training dataset contains 20,631 observations from
    100 simulated engines, with 21 sensor measurements and
    3 operational settings.

    The dataset represents simulated engine degradation
    over operating cycles. It is a research dataset, not
    live factory equipment data.
    """,

    # RUL prediction
    """
    IndusMind uses a Gradient Boosting Regressor to predict
    Remaining Useful Life (RUL).

    RUL represents the estimated number of operating cycles
    remaining before an engine reaches its simulated end of life.

    The model uses operating cycle, operational settings
    and selected sensor measurements as input features.

    The RUL target is capped at 125 cycles during model training.
    Predictions are also constrained to the range of 0 to 125.
    """,

    # Evaluation metrics
    """
    IndusMind's capped RUL prediction model was evaluated
    on a held-out validation set.

    Mean Absolute Error (MAE): 10.80 cycles.
    Root Mean Squared Error (RMSE): 15.53 cycles.
    R-squared score: 0.8615.

    These metrics apply to the capped RUL prediction task.
    They should not be directly compared with results from
    an uncapped RUL prediction experiment.
    """,

    # Feature importance
    """
    IndusMind Gradient Boosting model feature importance:

    sensor_11: 32.44 percent.
    cycle: 23.90 percent.
    sensor_4: 16.85 percent.
    sensor_12: 7.87 percent.
    sensor_9: 5.23 percent.
    sensor_7: 4.08 percent.
    sensor_15: 3.23 percent.
    sensor_20: 1.58 percent.
    sensor_14: 1.36 percent.
    sensor_2: 1.06 percent.
    sensor_21: 1.00 percent.
    sensor_3: 0.54 percent.
    sensor_17: 0.41 percent.
    sensor_13: 0.30 percent.
    sensor_8: 0.13 percent.

    The remaining operational settings and sensor features
    have very low or zero importance in this trained model.

    Sensor 11 is the most influential feature in this model.
    Feature importance describes predictive contribution,
    not physical causation or proof of mechanical failure.
    """,

    # Selected sensors
    """
    IndusMind uses 15 variable sensor measurements:

    sensor_2, sensor_3, sensor_4, sensor_6, sensor_7,
    sensor_8, sensor_9, sensor_11, sensor_12, sensor_13,
    sensor_14, sensor_15, sensor_17, sensor_20 and sensor_21.

    These features were selected after removing six
    practically constant sensors from the dataset.
    """,

    # Anomaly detection
    """
    IndusMind uses Isolation Forest for anomaly detection.

    The model is trained on sensor observations from
    training engines with RUL greater than 125 cycles,
    representing the selected relatively healthy operating
    portion of the training data.

    Isolation Forest identifies unusual sensor patterns
    compared with learned normal patterns.

    An anomaly does not automatically confirm mechanical
    failure. It indicates that further investigation may
    be appropriate.
    """,

    # General predictive maintenance
    """
    Predictive maintenance uses equipment data and analytics
    to identify potential degradation before equipment failure.

    It can support maintenance planning and reduce unexpected
    downtime, but model predictions should be reviewed by
    qualified maintenance personnel.
    """,

    # Sensor interpretation
    """
    Engine degradation can be associated with changing
    temperature, pressure, vibration and other sensor readings.

    Multiple sensor trends across operating cycles can provide
    more useful context than an isolated measurement.

    Sensor readings must be interpreted alongside operating
    conditions and historical behaviour.
    """,

    # Maintenance recommendations
    """
    When an unusual sensor pattern is detected, maintenance
    personnel may review sensor history, compare operating
    conditions, inspect relevant components and follow
    manufacturer maintenance procedures.

    An AI assistant should provide investigation suggestions,
    not claim that a specific component has failed without
    supporting evidence.
    """,

    # RAG architecture
    """
    IndusMind uses Retrieval-Augmented Generation (RAG).

    Maintenance documentation and project knowledge are
    converted into vector embeddings using the
    Sentence Transformers all-MiniLM-L6-v2 model.

    ChromaDB stores and retrieves relevant documents through
    semantic similarity search.

    The retrieved context and live engine information are
    supplied to the Llama 3.2 language model through Ollama.

    The language model generates explanations grounded in
    retrieved knowledge and the supplied engine information.
    """,

    # Project limitations
    """
    IndusMind is an educational predictive maintenance
    portfolio prototype.

    It uses simulated NASA C-MAPSS data, not real-time
    industrial sensor feeds.

    Its predictions are not certified maintenance decisions.
    Anomaly detection does not establish mechanical failure.
    The maintenance knowledge base contains general starter
    documentation and project-specific information, not
    official manufacturer maintenance manuals.
    """
]

# --------------------------------------------------
# 3. GENERATE EMBEDDINGS AND STORE DOCUMENTS
# --------------------------------------------------

print("Creating document embeddings...")

embeddings = embedding_model.encode(
    documents,
    show_progress_bar=True
).tolist()

collection.add(
    ids=[f"indusmind_doc_{i}" for i in range(len(documents))],
    documents=documents,
    embeddings=embeddings
)

print("IndusMind knowledge base ready!")
print("Documents stored:", collection.count())


# --------------------------------------------------
# 4. RETRIEVE RELEVANT DOCUMENTS
# --------------------------------------------------

def retrieve_context(question, n_results=4):

    query_embedding = embedding_model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(n_results, collection.count())
    )

    return "\n\n".join(results["documents"][0])


# --------------------------------------------------
# 5. GENERATE AI RESPONSE
# --------------------------------------------------

def ask_maintenance_copilot(question, engine_context=""):

    retrieved_context = retrieve_context(question)

    prompt = f"""
You are IndusMind, an AI-powered predictive maintenance
assistant.

You have access to project-specific knowledge, maintenance
documentation and live engine information.

INSTRUCTIONS:

1. Answer using the retrieved documents and supplied
   engine information.
2. Use the exact model metrics and feature importance
   values provided in the context.
3. Never invent sensor readings, predictions or metrics.
4. Clearly distinguish an anomaly from confirmed failure.
5. Explain technical concepts in simple language.
6. Give practical inspection suggestions when relevant.
7. If information is unavailable, say so honestly.
8. Never claim the system uses real-time factory data.

PROJECT AND RETRIEVED KNOWLEDGE:
{retrieved_context}

CURRENT ENGINE INFORMATION:
{engine_context}

USER QUESTION:
{question}

Provide a clear, structured answer.
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# --------------------------------------------------
# 6. TEST THE KNOWLEDGE BASE
# --------------------------------------------------

if __name__ == "__main__":

    question = (
        "Which sensor is most important in the "
        "IndusMind RUL prediction model?"
    )

    print("\nQuestion:", question)

    print("\nRetrieved context:")
    print(retrieve_context(question))

    print("\nAI Response:")
    print(ask_maintenance_copilot(question))
