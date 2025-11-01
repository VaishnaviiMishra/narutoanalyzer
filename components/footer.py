import gradio as gr

def create_footer():
    with gr.Row(elem_classes="footer"):
        with gr.Column():
            gr.HTML("""
            <div style="text-align: center; padding: 20px; background: #2c3e50; color: white; border-radius: 10px; margin-top: 30px;">
                <p style="margin: 10px 0;">© 2024 Naruto TV Analyzer</p>
                <p style="margin: 10px 0;">Models Used : facebook/bart-large-mnli | spacy en_core_web_trf | distilbert/distilbert-base-uncased | google/gemma-2b </p>    
                <p style="margin: 5px 0; font-size: 0.9em;">Naruto and all related characters are trademarks of Masashi Kishimoto/Shueisha</p>
                <p style="margin: 5px 0; font-size: 0.9em;">This is a fan-made analytical tool for educational purposes</p>
            </div>
            """)