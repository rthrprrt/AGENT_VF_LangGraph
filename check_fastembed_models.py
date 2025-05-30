# check_fastembed_models.py
from fastembed import TextEmbedding

print("Modèles TextEmbedding supportés par fastembed:")
supported_models = TextEmbedding.list_supported_models()
for model_info in supported_models:
    print(model_info)

# Alternative pour chercher un modèle spécifique si la liste est longue
# model_name_to_find = "bge-small-en-v1.5"
# found = any(model_name_to_find in str(model_info).lower() for model_info in supported_models)
# if found:
#     print(f"\nLe modèle contenant '{model_name_to_find}' semble être disponible.")
# else:
#     print(f"\nLe modèle contenant '{model_name_to_find}' n'a pas été trouvé directement.")