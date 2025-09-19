#!/usr/bin/env python3
"""
Interface web Streamlit pour l'OCR avec raisonnement
"""

import streamlit as st
import sys
import os
import json
import tempfile
from pathlib import Path
from PIL import Image
import io

# Ajouter le répertoire parent au path pour importer les modules
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Changer le répertoire de travail
os.chdir(root_dir)

try:
    from core.pipeline import pipeline
    from config.settings import settings
    PIPELINE_AVAILABLE = True
except ImportError as e:
    st.error(f"Erreur d'import: {e}")
    st.error("Assurez-vous de lancer l'application depuis le répertoire racine du projet")
    st.code("cd /home/asidimoh/pfe/NEW-OCR && streamlit run app/streamlit_app.py")
    PIPELINE_AVAILABLE = False

def main():
    st.set_page_config(
        page_title="OCR avec Raisonnement",
        page_icon="🔍",
        layout="wide"
    )
    
    if not PIPELINE_AVAILABLE:
        st.stop()
    
    st.title("🔍 OCR avec Raisonnement")
    st.markdown("Extrayez du texte et des informations structurées de vos documents")
    
    # Sidebar pour la configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Affichage des modèles configurés
        st.subheader("Modèles")
        st.text(f"OCR: {settings.OCR_MODEL}")
        st.text(f"Layout: {settings.LAYOUT_MODEL}")
        st.text(f"Reasoning: {settings.REASONING_MODEL}")
        st.text(f"Device: {settings.DEVICE}")
        
        # Options de traitement
        st.subheader("Options")
        use_layout = st.checkbox("Utiliser l'analyse de layout", value=True)
        format_type = st.selectbox("Format de sortie", ["plain", "markdown", "json"])
        
        # Gestion mémoire
        if st.button("📊 Info Mémoire"):
            try:
                memory_info = pipeline.get_memory_info()
                st.metric("Mémoire utilisée", f"{memory_info['current_memory_mb']:.1f} MB")
                st.metric("Usage", f"{memory_info['memory_usage_percent']:.1f}%")
            except Exception as e:
                st.error(f"Erreur mémoire: {e}")
        
        if st.button("🗑️ Décharger modèles"):
            try:
                pipeline.unload_models()
                st.success("Modèles déchargés")
            except Exception as e:
                st.error(f"Erreur: {e}")
    
    # Interface principale
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Document à traiter")
        
        # Upload de fichier
        uploaded_file = st.file_uploader(
            "Choisissez un fichier",
            type=['png', 'jpg', 'jpeg', 'pdf', 'bmp', 'tiff'],
            help="Formats supportés: PNG, JPG, JPEG, PDF, BMP, TIFF"
        )
        
        # Prompt utilisateur
        user_prompt = st.text_area(
            "Prompt d'extraction (optionnel)",
            placeholder="Ex: Extraire le nom, l'âge, l'email et le téléphone",
            help="Décrivez quelles informations vous voulez extraire"
        )
        
        # Exemples de prompts
        st.subheader("💡 Exemples de prompts")
        example_prompts = [
            "Extraire toutes les informations personnelles",
            "Extraire le montant, la date et le numéro de facture",
            "Extraire les compétences et l'expérience professionnelle",
            "Extraire les données de contact (nom, téléphone, email)",
            "Extraire les montants et dates importantes"
        ]
        
        for prompt in example_prompts:
            if st.button(f"📝 {prompt}", key=prompt):
                user_prompt = prompt
                st.rerun()
        
        # Bouton de traitement
        process_button = st.button("🚀 Traiter le document", type="primary", disabled=uploaded_file is None)
    
    with col2:
        st.header("📊 Résultats")
        
        if uploaded_file and process_button:
            # Afficher l'image uploadée
            if uploaded_file.type.startswith('image/'):
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Document à traiter", use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors du chargement de l'image: {e}")
            
            # Traitement
            with st.spinner("Traitement en cours..."):
                try:
                    # Sauvegarder le fichier temporairement
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Traitement avec la pipeline
                    result = pipeline.process(
                        input_source=tmp_path,
                        user_prompt=user_prompt if user_prompt.strip() else None,
                        use_layout=use_layout,
                        format_type=format_type,
                        output_format="json"
                    )
                    
                    # Nettoyer le fichier temporaire
                    os.unlink(tmp_path)
                    
                    # Afficher les résultats
                    if result['success']:
                        st.success("✅ Traitement réussi !")
                        
                        # Onglets pour différents types de résultats
                        tab1, tab2, tab3 = st.tabs(["📝 Texte extrait", "🏗️ Données structurées", "ℹ️ Métadonnées"])
                        
                        with tab1:
                            st.subheader("Texte extrait")
                            st.text_area("", value=result.get('text', ''), height=300, disabled=True)
                            
                            # Bouton de téléchargement du texte
                            st.download_button(
                                label="📥 Télécharger le texte",
                                data=result.get('text', ''),
                                file_name=f"texte_extrait_{uploaded_file.name}.txt",
                                mime="text/plain"
                            )
                        
                        with tab2:
                            st.subheader("Données structurées")
                            structured_data = result.get('structured_data')
                            if structured_data:
                                st.json(structured_data)
                                
                                # Bouton de téléchargement JSON
                                st.download_button(
                                    label="📥 Télécharger JSON",
                                    data=json.dumps(structured_data, indent=2, ensure_ascii=False),
                                    file_name=f"donnees_structurees_{uploaded_file.name}.json",
                                    mime="application/json"
                                )
                            else:
                                st.info("Aucune donnée structurée extraite. Utilisez un prompt pour extraire des informations spécifiques.")
                        
                        with tab3:
                            st.subheader("Métadonnées")
                            metadata = {
                                "Pages traitées": result.get('pages_processed', 0),
                                "Temps de traitement": f"{result.get('processing_time', 0):.2f}s",
                                "Modèle OCR": settings.OCR_MODEL,
                                "Layout utilisé": use_layout,
                                "Format": format_type
                            }
                            
                            for key, value in metadata.items():
                                st.metric(key, value)
                    
                    else:
                        st.error(f"❌ Erreur lors du traitement: {result.get('error', 'Erreur inconnue')}")
                
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
        
        elif not uploaded_file:
            st.info("👆 Uploadez un document pour commencer")

if __name__ == "__main__":
    main()