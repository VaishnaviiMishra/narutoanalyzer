import gradio as gr
import os
import sys
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Suppress known deprecation warnings from transformers library
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


print("ℹ️ Using pre-computed data from stubs/ folder (spaCy model not required)")

# Now import modules that depend on spaCy
from theme_classifier import ThemeClassifier
from character_network import NamedEntityRecognizer, CharacterNetworkGenerator
from text_classification import JutsuClassifier
from character_chatbot import GeminiChatBot
from components import create_navbar, create_hero_section, create_footer, create_about_section

load_dotenv()

# Get project root directory (works on any OS - Windows, Linux, Mac)
PROJECT_ROOT = Path(__file__).parent.absolute()
STUBS_DIR = PROJECT_ROOT / "stubs"
THEME_OUTPUT_PATH = STUBS_DIR / "theme_classifier_output.csv"
NER_OUTPUT_PATH = STUBS_DIR / "ner_output.csv"

# Initialize the Gemini chatbot
try:
    character_chatbot = GeminiChatBot()
    chatbot_available = character_chatbot.available
    if chatbot_available:
        print("Gemini chatbot initialized successfully!")
    else:
        print("Gemini chatbot failed to initialize")
except Exception as e:
    print(f"Error initializing chatbot: {e}")
    chatbot_available = False
    character_chatbot = None

def get_themes(theme_list_str, subtitles_path, save_path):
    # Handle save path - if it's just a filename, use stubs directory
    if save_path and not ('/' in save_path or '\\' in save_path):
        save_path = str(STUBS_DIR / save_path)
    
    theme_list = theme_list_str.split(',')
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path, save_path)
    theme_list = [theme for theme in theme_list if theme != 'dialogue']
    output_df = output_df[theme_list]
    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ['Theme','Score']
    output_chart = gr.BarPlot(
        output_df,
        x="Theme",
        y="Score",
        title="Series Themes",
        tooltip=["Theme","Score"],
        vertical=False,
        width=500,
        height=260
    )
    return output_chart

def get_character_network(subtitles_path, ner_path):
    # Handle NER path - if it's just a filename, use stubs directory
    if ner_path and not ('/' in ner_path or '\\' in ner_path):
        ner_path = str(STUBS_DIR / ner_path)
    
    # This will either process subtitles with spaCy OR load existing NER data
    ner = NamedEntityRecognizer()
    ner_df = ner.get_ners(subtitles_path, ner_path)
    
    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)
    return html

def classify_text(text_classifcation_model, text_classifcation_data_path, text_to_classify):
    jutsu_classifier = JutsuClassifier(
        model_path=text_classifcation_model,
        data_path=text_classifcation_data_path,
        huggingface_token=os.getenv('huggingface_token')
    )
    output = jutsu_classifier.classify_jutsu(text_to_classify)
    return output[0]

def main():
    # Use Gradio's built-in theme instead of custom CSS for HF Spaces compatibility
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="red",
        neutral_hue="slate",
        font=("Arial", "sans-serif")
    )
    
    with gr.Blocks(title="Naruto TV Analyzer", theme=theme) as iface:
        
        # Navigation Bar
        create_navbar()
        
        # Hero Section
        create_hero_section()

        # About Section
        create_about_section()
        
        # Theme Classification Section
        with gr.Row(elem_id="theme-section", elem_classes="section"):
            with gr.Column():
                gr.HTML("<h2 style='color: #FF8C00 !important;'>Theme Classification</h2>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot()
                    with gr.Column():
                        theme_list = gr.Textbox(
                            label="Themes (comma-separated)", 
                            value="friendship,hope,sacrifice,battle,self development,betrayal,love,dialogue"
                        )
                        # Commented out for deployment - using pre-computed data
                        # Uncomment below if you want users to provide their own subtitle files
                        # subtitles_path = gr.Textbox(
                        #     label="Subtitles or Script Path", 
                        #     value="  "
                        # )
                        save_path = gr.Textbox(
                            label="Save Path", 
                            value="theme_classifier_output.csv",
                            placeholder="Filename (saved in stubs/)"
                        )
                        get_themes_button = gr.Button("Analyze Themes", variant="primary")
                        # Using empty string for subtitles_path since we're using pre-computed data
                        get_themes_button.click(get_themes, inputs=[theme_list, gr.Textbox(value="", visible=False), save_path], outputs=[plot])

        # Character Network Section
        with gr.Row(elem_id="network-section", elem_classes="section"):
            with gr.Column():
                gr.HTML("<h2 style='color: #FF8C00 !important;'>Character Network</h2>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        # Commented out for deployment - using pre-computed data
                        # Uncomment below if you want users to provide their own subtitle files
                        # subtitles_path_ner = gr.Textbox(
                        #     label="Subtitles or Script Path", 
                        #     value=" "
                        # )
                        ner_path = gr.Textbox(
                            label="NERs Save Path", 
                            value="ner_output.csv",
                            placeholder="Filename (saved in stubs/)"
                        )
                        get_network_graph_button = gr.Button("Generate Network", variant="primary")
                        # Using empty string for subtitles_path_ner since we're using pre-computed data
                        get_network_graph_button.click(get_character_network, inputs=[gr.Textbox(value="", visible=False), ner_path], outputs=[network_html])

        # Text Classification with LLMs
        with gr.Row(elem_id="jutsu-section", elem_classes="section"):
            with gr.Column():
                gr.HTML("<h2 style='color: #FF8C00 !important;'>Jutsu Classification</h2>")
                with gr.Row():
                    with gr.Column():
                        text_classification_output = gr.Textbox(label="Classification Result", lines=3)
                    with gr.Column():
                        text_classifcation_model = gr.Textbox(
                            label='Model Path', 
                            value="vaishnaviiii34/jutsu_classifier"
                        )
                        # Commented out for deployment - model handles data internally
                        # Uncomment below if you want users to provide custom data path
                        # text_classifcation_data_path = gr.Textbox(
                        #     label='Data Path', 
                        #     value="  "
                        # )
                        text_to_classify = gr.Textbox(
                            label='Jutsu Description', 
                            value="The Rasengan is a powerful, spherical ball of rapidly spinning chakra created and held in the user's palm, developed by Minato Namikaze and described as the pinnacle of shape transformation in Naruto. It's a high-level, A-rank technique that focuses on compressing and rotating chakra to achieve incredible density and rotational power",
                            lines=3
                        )
                        classify_text_button = gr.Button("Classify Jutsu", variant="primary")
                        # Using empty string for data path since model is pre-trained
                        classify_text_button.click(classify_text, inputs=[text_classifcation_model, gr.Textbox(value="", visible=False), text_to_classify], outputs=[text_classification_output])

       # Character Chatbot Section
        with gr.Row(elem_id="chat-section", elem_classes="section"):
            with gr.Column():
                gr.HTML("<h2 style='color: #FF8C00 !important;'>Chat with Naruto Characters</h2>")
                if not chatbot_available:
                    gr.Warning("Please add GEMINI_API_KEY=your_key_here to your .env file")
                else:
                    gr.Info("Chat with Naruto, Sasuke, or Sakura!")
                
                # Separate chat interfaces for each character
                with gr.Tab("Naruto"):
                    def chat_naruto(message, history):
                        return character_chatbot.chat(message, history, "naruto")
                    
                    gr.ChatInterface(
                        chat_naruto,
                        examples=[
                            "What's your dream?",
                            "Tell me about your abilities",
                            "Who is your sensei?",
                            "What's your strongest technique?"
                        ],
                        title="Chat with Naruto"
                    )
                
                with gr.Tab("Sasuke"):
                    def chat_sasuke(message, history):
                        return character_chatbot.chat(message, history, "sasuke")
                    
                    gr.ChatInterface(
                        chat_sasuke,
                        examples=[
                            "What is your goal?",
                            "Tell me about the Uchiha clan",
                            "Why do you seek power?",
                            "What is the Sharingan?"
                        ],
                        title="Chat with Sasuke"
                    )
                
                with gr.Tab("Sakura"):
                    def chat_sakura(message, history):
                        return character_chatbot.chat(message, history, "sakura")
                    
                    gr.ChatInterface(
                        chat_sakura,
                        examples=[
                            "What medical jutsu do you know?",
                            "How did you train with Tsunade?",
                            "What is chakra control?",
                            "Tell me about your strength"
                        ],
                        title="Chat with Sakura"
                    )


        create_footer() 

        # Launch with proper settings for deployment platforms
        # Get port from environment variable (used by Render, Heroku, etc.)
        port = int(os.getenv("PORT", 7860))
        
        print(f"\n{'='*60}")
        print(f"🚀 Naruto TV Analyzer is starting...")
        print(f"{'='*60}")
        print(f"📍 Local URL:   http://localhost:{port}")
        print(f"📍 Network URL: http://127.0.0.1:{port}")
        print(f"{'='*60}\n")
        
        iface.launch(
            server_name="0.0.0.0",  # Bind to all interfaces for deployment
            server_port=port,  # Use PORT from environment or default to 7860
            share=False,
            inbrowser=False  # Don't auto-open browser on server
        )

if __name__ == '__main__':
    main()