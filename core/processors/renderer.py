import json
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class Renderer:
    """Rendu des resultats en differents formats"""
    
    @staticmethod
    def render_to_json(result_dict: Dict[str, Any], pretty: bool = True) -> str:
        """Rend le resultat en JSON"""
        try:
            if pretty:
                return json.dumps(result_dict, ensure_ascii=False, indent=2)
            else:
                return json.dumps(result_dict, ensure_ascii=False)
        except Exception as e:
            logger.error(f"JSON rendering failed: {e}")
            return '{"error": "JSON rendering failed"}'
    
    @staticmethod
    def render_to_html(text: str, structured_data: Optional[Dict[str, Any]] = None) -> str:
        """Rend le resultat en HTML simple"""
        try:
            html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>OCR Result</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .text-content {{ background: #f5f5f5; padding: 15px; margin: 10px 0; white-space: pre-wrap; }}
        .structured-data {{ background: #e8f4fd; padding: 15px; margin: 10px 0; }}
        pre {{ background: #f8f8f8; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>Resultat OCR</h1>
    
    <h2>Texte extrait</h2>
    <div class="text-content">{text}</div>
"""
            
            if structured_data:
                html += f"""
    <h2>Donnees structurees</h2>
    <div class="structured-data">
        <pre>{json.dumps(structured_data, ensure_ascii=False, indent=2)}</pre>
    </div>
"""
            
            html += """
</body>
</html>"""
            
            return html
            
        except Exception as e:
            logger.error(f"HTML rendering failed: {e}")
            return f"<html><body><h1>Erreur de rendu</h1><p>{e}</p></body></html>"
    
    @staticmethod
    def render_to_markdown(text: str, structured_data: Optional[Dict[str, Any]] = None) -> str:
        """Rend le resultat en Markdown"""
        try:
            markdown = f"""# Resultat OCR

## Texte extrait

```
{text}
```
"""
            
            if structured_data:
                markdown += f"""
## Donnees structurees

```json
{json.dumps(structured_data, ensure_ascii=False, indent=2)}
```
"""
            
            return markdown
            
        except Exception as e:
            logger.error(f"Markdown rendering failed: {e}")
            return f"# Erreur de rendu\n\n{e}"