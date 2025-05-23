# src/nodes/n2_journal_ingestor_anonymizer.py
import os
import shutil
from typing import List, Dict, Any, TypedDict, Optional
from langchain_community.document_loaders import TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Supposons que AgentState et settings soient définis ailleurs
# from src.state import AgentState
# from src.config import settings # Pour settings.embedding_model_name

# Pour l'exemple, définissons une structure simplifiée pour AgentState et settings
class AgentState(TypedDict):
    journal_path: str
    vector_store_path: str
    raw_journal_entries: List[Dict[str, Any]] # Sera peuplé par ce noeud, mais le texte sera anonymisé par LLM-Reasoning-Foundations
    recreate_vector_store: Optional[bool]
    vector_store_initialized: Optional[bool]
    # ... autres champs d'état

class Settings: # Simulation
    embedding_model_name: str = "fastembed/BAAI/bge-small-en-v1.5" # Exemple de valeur

settings = Settings() # Instance simulée

def load_raw_journal_entries(journal_path: str) -> List[Dict[str, Any]]:
    """Charge le contenu brut des fichiers .txt et .docx."""
    entries = []
    if not os.path.isdir(journal_path):
        print(f"Error: Journal path {journal_path} not found or not a directory.")
        return entries

    for filename in os.listdir(journal_path):
        file_path = os.path.join(journal_path, filename)
        entry_date = None # Logique d'extraction de date basique (ex: du nom de fichier)
        # Exemple: if re.match(r'\d{4}-\d{2}-\d{2}', filename): entry_date = filename.split('.')[0]

        raw_text_content = ""
        try:
            if filename.endswith(".txt"):
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                if docs:
                    raw_text_content = docs[0].page_content
            elif filename.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)
                docs = loader.load()
                if docs:
                    raw_text_content = docs[0].page_content
            else:
                continue # Ignorer les autres types de fichiers

            entries.append({
                'source_document': filename, # Garder le nom du fichier original
                'date': entry_date,
                'text': raw_text_content # Ce texte sera ensuite anonymisé par LLM-Reasoning-Foundations
            })
        except Exception as e:
            print(f"Error loading file {filename}: {e}")
            
    return entries

def _chunk_text(text_entries: List[Dict[str, Any]]) -> List[Document]:
    """Applique le chunking avancé et prépare les documents pour l'embedding."""
    # Paramètres de chunking définis dans le plan
    chunk_size = 1500
    chunk_overlap = 200
    # Séparateurs par défaut: ["\n\n", "\n", " ", ""]
    # Considérer l'ajout de "# ", "## " si pertinent pour le format du journal

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    all_docs_for_faiss = []
    for entry_index, entry in enumerate(text_entries):
        # IMPORTANT: Supposons ici que 'entry["text"]' est le texte DÉJÀ anonymisé
        # fourni par LLM-Reasoning-Foundations après son traitement.
        # LLM-Reasoning-Foundations pourrait mettre à jour 'raw_journal_entries'
        # ou ajouter une nouvelle clé comme 'anonymized_text'.
        # Pour cette fonction, nous nous attendons à ce que le texte soit prêt.
        
        text_to_chunk = entry.get("anonymized_text", entry.get("text", "")) # Prioriser le texte anonymisé

        if not text_to_chunk:
            continue

        chunks = text_splitter.split_text(text_to_chunk)
        
        for i, chunk_text in enumerate(chunks):
            metadata = {
                'source_document': entry.get('source_document', f"unknown_doc_{entry_index}"),
                'journal_date': entry.get('date', None),
                'chunk_index': i,
                'chunk_id': f"{entry.get('source_document', f'doc_{entry_index}')}_chunk_{i}" # ID unique
            }
            all_docs_for_faiss.append(Document(page_content=chunk_text, metadata=metadata))
            
    return all_docs_for_faiss

def journal_ingestor_anonymizer_node(state: AgentState) -> Dict[str, Any]:
    """
    Nœud responsable de l'ingestion, du chunking, de l'embedding et de la création/gestion du vector store FAISS.
    L'anonymisation et la gestion du ton sont gérées en amont ou par un autre spécialiste au sein de ce nœud.
    Ce spécialiste (LLM-RAG-Specialist) se concentre sur la partie RAG après l'anonymisation.
    """
    print("--- INGÉSTION ET VECTORISATION DU JOURNAL ---")
    journal_path = state.get("journal_path")
    vector_store_path = state.get("vector_store_path")
    recreate_vector_store = state.get("recreate_vector_store", False)

    if not journal_path or not vector_store_path:
        print("Error: journal_path or vector_store_path not defined in state.")
        return {"vector_store_initialized": False, "raw_journal_entries": []}

    # 1. Charger les entrées brutes (LLM-RAG-Specialist)
    # Note: LLM-Reasoning-Foundations interviendra ici pour anonymiser/modifier le ton
    # sur 'state.raw_journal_entries' avant le chunking et l'embedding.
    # Pour l'instant, cette fonction charge juste les données.
    # L'étape d'anonymisation par LLM-Reasoning-Foundations est supposée se produire
    # sur le champ 'text' de chaque dictionnaire dans raw_journal_entries.
    raw_journal_entries = load_raw_journal_entries(journal_path)
    if not raw_journal_entries:
        print("No journal entries loaded.")
        return {"vector_store_initialized": False, "raw_journal_entries": raw_journal_entries}
    
    # Mettre à jour l'état avec les entrées brutes (sera utilisé par LLM-Reasoning-Foundations)
    current_state_update: Dict[str, Any] = {"raw_journal_entries": raw_journal_entries}

    # --- Point de collaboration avec LLM-Reasoning-Foundations ---
    # LLM-Reasoning-Foundations aura modifié 'raw_journal_entries' (par exemple, en ajoutant
    # une clé 'anonymized_text' ou en modifiant 'text' in-place).
    # Pour la suite, nous utiliserons le texte traité.

    # 2. Chunker le texte (après anonymisation par LLM-Reasoning-Foundations)
    # Utiliser 'raw_journal_entries' qui devrait maintenant contenir le texte anonymisé
    # dans le champ 'text' ou un champ dédié comme 'anonymized_text'.
    print("Chunking des textes anonymisés...")
    documents_for_faiss = _chunk_text(raw_journal_entries) # Utilise les entrées potentiellement modifiées

    if not documents_for_faiss:
        print("No documents to add to FAISS after chunking.")
        current_state_update["vector_store_initialized"] = False
        return current_state_update

    # 3. & 4. Générer Embeddings et Créer/Peupler FAISS (LLM-RAG-Specialist)
    print(f"Utilisation du modèle d'embedding: {settings.embedding_model_name}")
    embeddings = FastEmbedEmbeddings(model_name=settings.embedding_model_name)

    vector_store_dir = vector_store_path
    
    if recreate_vector_store and os.path.exists(vector_store_dir):
        print(f"Suppression du vector store existant à: {vector_store_dir}")
        shutil.rmtree(vector_store_dir) # Supprime le répertoire et son contenu

    if not os.path.exists(os.path.join(vector_store_dir, "index.faiss")) or recreate_vector_store:
        if not documents_for_faiss:
            print("Aucun document à indexer, le vector store ne sera pas créé.")
            current_state_update["vector_store_initialized"] = False
            return current_state_update
        print("Création d'un nouveau vector store FAISS...")
        try:
            db = FAISS.from_documents(documents_for_faiss, embeddings)
            os.makedirs(vector_store_dir, exist_ok=True)
            db.save_local(vector_store_dir)
            print(f"Vector store FAISS créé et sauvegardé à: {vector_store_dir}")
            current_state_update["vector_store_initialized"] = True
        except Exception as e:
            print(f"Erreur lors de la création du vector store FAISS: {e}")
            current_state_update["vector_store_initialized"] = False
            return current_state_update
    else:
        print(f"Utilisation du vector store FAISS existant à: {vector_store_dir}")
        # On pourrait ajouter une logique de chargement ici si nécessaire, mais pour Phase 1,
        # on suppose que l'existence + recreate_vector_store=False signifie qu'il est prêt.
        current_state_update["vector_store_initialized"] = True
        
    return current_state_update