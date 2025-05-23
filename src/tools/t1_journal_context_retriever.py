# src/tools/t1_journal_context_retriever.py
from typing import List, Dict, Any, Type, Optional
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Supposons que AgentState et settings soient définis ailleurs
# from src.state import AgentState
# from src.config import settings

class JournalContextRetrieverArgs(BaseModel):
    query_or_keywords: str = Field(description="La requête ou les mots-clés à rechercher dans le journal.")
    k_retrieval_count: Optional[int] = Field(default=3, description="Nombre de chunks pertinents à récupérer (défaut 3).")

class JournalContextRetrieverTool(BaseTool):
    name: str = "journal_context_retriever"
    description: str = (
        "Récupère des extraits pertinents et anonymisés du journal d'apprentissage de l'étudiant "
        "en fonction d'une requête ou de mots-clés. Utiliser pour trouver des expériences spécifiques, "
        "des tâches, des réflexions ou des détails mentionnés dans le journal."
    )
    args_schema: Type[BaseModel] = JournalContextRetrieverArgs
    
    # Ces attributs seront configurés par le nœud qui instancie cet outil
    vector_store_path: str 
    embedding_model_name: str 

    def _run(self, query_or_keywords: str, k_retrieval_count: Optional[int] = 3) -> List[Dict[str, Any]]:
        print(f"--- RÉCUPÉRATION DE CONTEXTE DU JOURNAL POUR : '{query_or_keywords}' ---")
        
        effective_k = k_retrieval_count if k_retrieval_count is not None else 3

        try:
            embeddings = FastEmbedEmbeddings(model_name=self.embedding_model_name)
        except Exception as e:
            print(f"Erreur lors de l'initialisation du modèle d'embedding ({self.embedding_model_name}): {e}")
            return [{"error": "Échec de l'initialisation du modèle d'embedding", "details": str(e)}]

        if not os.path.exists(self.vector_store_path) or \
           not os.path.exists(os.path.join(self.vector_store_path, "index.faiss")):
            error_msg = f"Vector store non trouvé à {self.vector_store_path}"
            print(f"Erreur: {error_msg}")
            return [{"error": error_msg}]
        
        try:
            # allow_dangerous_deserialization est nécessaire pour FAISS avec LangChain >= 0.1.10
            vector_store = FAISS.load_local(
                self.vector_store_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"Erreur lors du chargement de l'index FAISS depuis {self.vector_store_path}: {e}")
            return [{"error": "Échec du chargement de l'index FAISS", "details": str(e)}]

        try:
            retrieved_docs_with_scores = vector_store.similarity_search_with_score(
                query_or_keywords, k=effective_k
            )
        except Exception as e:
            print(f"Erreur lors de la recherche de similarité : {e}")
            return [{"error": "Erreur lors de la recherche de similarité", "details": str(e)}]

        output_excerpts = []
        if not retrieved_docs_with_scores:
            print("Aucun résultat pertinent trouvé pour la requête.")
            return [] 

        for doc, score in retrieved_docs_with_scores:
            output_excerpts.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score) # Score de distance L2 (plus petit = mieux) pour FAISS par défaut
            })
        
        print(f"Récupéré {len(output_excerpts)} extraits.")
        return output_excerpts

    async def _arun(self, query_or_keywords: str, k_retrieval_count: Optional[int] = 3) -> List[Dict[str, Any]]:
        # Pour l'instant, la version asynchrone appelle la version synchrone.
        # Pourrait être optimisée avec des bibliothèques asynchrones si FAISS/FastEmbed le supportent bien.
        return self._run(query_or_keywords=query_or_keywords, k_retrieval_count=k_retrieval_count)

# Note: L'instanciation de cet outil se fera dans le nœud LangGraph qui l'utilise, 
# par exemple N5_ContextRetrievalNode. C'est ce nœud qui fournira 
# `vector_store_path` et `embedding_model_name` à partir de l'état de l'agent et des settings.
# Exemple d'instanciation (ne pas mettre ici, mais dans le nœud utilisateur):
# tool_instance = JournalContextRetrieverTool(
#     vector_store_path=state.vector_store_path, 
#     embedding_model_name=settings.embedding_model_name
# )