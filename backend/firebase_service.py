import os
import logging
from typing import List, Dict, Any, Optional
from backend.config import settings
from backend.database import local_db

logger = logging.getLogger("FirebaseService")
logger.setLevel(logging.INFO)

_firebase_initialized = False
_firebase_disabled = False
_db_client = None

def init_firebase():
    global _firebase_initialized, _firebase_disabled, _db_client
    if _firebase_disabled:
        return None
    if _firebase_initialized:
        return _db_client

    key_path = getattr(settings, "SERVICE_ACCOUNT_KEY_PATH", "serviceAccountKey.json")
    
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            if os.path.exists(key_path):
                with open(key_path, "r") as f:
                    content = f.read()
                    if "your-firebase-project-id" in content or "YOUR_PRIVATE_KEY" in content:
                        logger.info("Placeholder serviceAccountKey.json detected. Using Local Database.")
                        _firebase_disabled = True
                        return None
                
                cred = credentials.Certificate(key_path)
                firebase_admin.initialize_app(cred)
                logger.info(f"Firebase Admin SDK initialized using {key_path}")
            elif settings.FIREBASE_PROJECT_ID and settings.FIREBASE_PRIVATE_KEY and "your-firebase-project-id" not in settings.FIREBASE_PROJECT_ID:
                private_key = settings.FIREBASE_PRIVATE_KEY.replace('\\n', '\n')
                cred_dict = {
                    "type": "service_account",
                    "project_id": settings.FIREBASE_PROJECT_ID,
                    "private_key_id": settings.FIREBASE_PRIVATE_KEY_ID,
                    "private_key": private_key,
                    "client_email": settings.FIREBASE_CLIENT_EMAIL,
                    "client_id": settings.FIREBASE_CLIENT_ID,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
                logger.info("Firebase Admin SDK initialized using environment variables")
            else:
                logger.info("No valid Firebase credentials found. Using Local Database.")
                _firebase_disabled = True
                return None

        client = firestore.client()
        # Test connection to verify JWT validity
        try:
            client.collection("health_check").limit(1).get()
            _db_client = client
            _firebase_initialized = True
            return _db_client
        except Exception as auth_err:
            logger.warning(f"Firebase Firestore authentication failed: {auth_err}. Falling back to Local Database.")
            _firebase_disabled = True
            _db_client = None
            return None

    except Exception as e:
        logger.warning(f"Firebase initialization failed: {e}. Falling back to Local Database.")
        _firebase_disabled = True
        _db_client = None
        return None

def is_firebase_active() -> bool:
    client = init_firebase()
    return client is not None

def disable_firebase():
    global _firebase_disabled, _db_client
    _firebase_disabled = True
    _db_client = None

# Unified Storage Interface functions

def create_task(task_data: Dict[str, Any]) -> str:
    client = init_firebase()
    task_id = task_data.get("task_id", "T000")
    if client:
        try:
            client.collection("tasks").document(task_id).set(task_data)
        except Exception as e:
            logger.warning(f"Firestore error creating task {task_id}: {e}. Disabling Firebase.")
            disable_firebase()
    local_db.set_document("tasks", task_id, task_data)
    return task_id

def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    client = init_firebase()
    if client:
        try:
            doc = client.collection("tasks").document(task_id).get()
            if doc.exists:
                return doc.to_dict()
        except Exception as e:
            logger.warning(f"Firestore error reading task {task_id}: {e}. Disabling Firebase.")
            disable_firebase()
    return local_db.get_document("tasks", task_id)

def get_tasks() -> List[Dict[str, Any]]:
    client = init_firebase()
    if client:
        try:
            docs = client.collection("tasks").stream()
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.warning(f"Firestore error getting tasks: {e}. Disabling Firebase.")
            disable_firebase()
    return local_db.get_all_documents("tasks")

def update_task(task_id: str, update_data: Dict[str, Any]) -> bool:
    client = init_firebase()
    if client:
        try:
            client.collection("tasks").document(task_id).update(update_data)
        except Exception as e:
            logger.warning(f"Firestore error updating task {task_id}: {e}. Disabling Firebase.")
            disable_firebase()
    return local_db.update_document("tasks", task_id, update_data)

def save_result(result_data: Dict[str, Any]) -> str:
    client = init_firebase()
    task_id = result_data.get("task_id", "R000")
    if client:
        try:
            client.collection("scheduling_results").document(task_id).set(result_data)
        except Exception as e:
            logger.warning(f"Firestore error saving result: {e}. Disabling Firebase.")
            disable_firebase()
    local_db.set_document("scheduling_results", task_id, result_data)
    return task_id

def get_results() -> List[Dict[str, Any]]:
    client = init_firebase()
    if client:
        try:
            docs = client.collection("scheduling_results").stream()
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.warning(f"Firestore error getting results: {e}. Disabling Firebase.")
            disable_firebase()
    return local_db.get_all_documents("scheduling_results")

def save_prediction(prediction_data: Dict[str, Any]) -> str:
    client = init_firebase()
    doc_id = str(prediction_data.get("timestamp", 0))
    if client:
        try:
            client.collection("predictions").document(doc_id).set(prediction_data)
        except Exception as e:
            logger.warning(f"Firestore error saving prediction: {e}. Disabling Firebase.")
            disable_firebase()
    local_db.set_document("predictions", doc_id, prediction_data)
    return doc_id

def save_violation(violation_data: Dict[str, Any]) -> str:
    client = init_firebase()
    v_id = violation_data.get("violation_id", "V000")
    if client:
        try:
            client.collection("isolation_violations").document(v_id).set(violation_data)
        except Exception as e:
            logger.warning(f"Firestore error saving violation: {e}. Disabling Firebase.")
            disable_firebase()
    local_db.set_document("isolation_violations", v_id, violation_data)
    return v_id

def get_violations() -> List[Dict[str, Any]]:
    client = init_firebase()
    if client:
        try:
            docs = client.collection("isolation_violations").stream()
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.warning(f"Firestore error getting violations: {e}. Disabling Firebase.")
            disable_firebase()
    return local_db.get_all_documents("isolation_violations")

def save_gpus(gpu_list: List[Dict[str, Any]]):
    client = init_firebase()
    for gpu in gpu_list:
        gpu_id = gpu.get("gpu_id")
        if client:
            try:
                client.collection("gpu_nodes").document(gpu_id).set(gpu)
            except Exception as e:
                logger.warning(f"Firestore error saving GPU {gpu_id}: {e}. Disabling Firebase.")
                disable_firebase()
        local_db.set_document("gpu_nodes", gpu_id, gpu)

def get_gpus() -> List[Dict[str, Any]]:
    client = init_firebase()
    if client:
        try:
            docs = client.collection("gpu_nodes").stream()
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.warning(f"Firestore error getting GPUs: {e}. Disabling Firebase.")
            disable_firebase()
    return local_db.get_all_documents("gpu_nodes")

def reset_simulation_store():
    client = init_firebase()
    collections_to_clear = ["tasks", "scheduling_results", "predictions", "isolation_violations", "metrics"]
    if client:
        try:
            for col in collections_to_clear:
                docs = client.collection(col).stream()
                for d in docs:
                    d.reference.delete()
        except Exception as e:
            logger.warning(f"Firestore error clearing collections: {e}. Disabling Firebase.")
            disable_firebase()
    local_db.clear_all()
