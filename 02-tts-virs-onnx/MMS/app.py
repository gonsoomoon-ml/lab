import gradio as gr
import librosa
from asr import transcribe, ASR_EXAMPLES, ASR_LANGUAGES, ASR_NOTE
from tts import synthesize, TTS_EXAMPLES, TTS_LANGUAGES
from lid import identify, LID_EXAMPLES



mms_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(),
        gr.Dropdown(
            [f"{k} ({v})" for k, v in ASR_LANGUAGES.items()],
            label="Language",
            value="eng English",
        ),
        # gr.Checkbox(label="Use Language Model (if available)", default=True),
    ],
    outputs="text",
    examples=ASR_EXAMPLES,
    title="Speech-to-text",
    description=(
        "Transcribe audio from a microphone or input file in your desired language."
    ),
    article=ASR_NOTE,
    allow_flagging="never",
)

mms_synthesize = gr.Interface(
    fn=synthesize,
    inputs=[
        gr.Text(label="Input text"),
        gr.Dropdown(
            [f"{k} ({v})" for k, v in TTS_LANGUAGES.items()],
            label="Language",
            value="eng English",
        ),
        gr.Slider(minimum=0.1, maximum=4.0, value=1.0, step=0.1, label="Speed"),
    ],
    outputs=[
        gr.Audio(label="Generated Audio", type="numpy"),
        gr.Text(label="Filtered text after removing OOVs"),
    ],
    examples=TTS_EXAMPLES,
    title="Text-to-speech",
    description=("Generate audio in your desired language from input text."),
    allow_flagging="never",
)

mms_identify = gr.Interface(
    fn=identify,
    inputs=[
        gr.Audio(),
    ],
    outputs=gr.Label(num_top_classes=10),
    examples=LID_EXAMPLES,
    title="Language Identification",
    description=("Identity the language of input audio."),
    allow_flagging="never",
)

tabbed_interface = gr.TabbedInterface(
    [mms_transcribe, mms_synthesize, mms_identify],
    ["Speech-to-text", "Text-to-speech", "Language Identification"],
)

with gr.Blocks() as demo:
    gr.Markdown(
        "<p align='center' style='font-size: 20px;'>MMS: Scaling Speech Technology to 1000+ languages demo. See our <a href='https://ai.facebook.com/blog/multilingual-model-speech-recognition/'>blog post</a> and <a href='https://arxiv.org/abs/2305.13516'>paper</a>.</p>"
    )
    gr.HTML(
        """<center>Click on the appropriate tab to explore Speech-to-text (ASR), Text-to-speech (TTS) and Language identification (LID) demos.   </center>"""
    )
    gr.HTML(
        """<center>You can also finetune MMS models on your data using the recipes provides here - <a href='https://huggingface.co/blog/mms_adapters'>ASR</a> <a href='https://github.com/ylacombe/finetune-hf-vits'>TTS</a>  </center>"""
    )
    gr.HTML(
        """<center><a href="https://huggingface.co/spaces/facebook/MMS?duplicate=true"  style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank"><img style="margin-bottom: 0em;display: inline;margin-top: -.25em;" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a> for more control and no queue.</center>"""
    )

    tabbed_interface.render()
    gr.HTML(
        """
            <div class="footer" style="text-align:center">
                <p>
                    Model by <a href="https://ai.facebook.com" style="text-decoration: underline;" target="_blank">Meta AI</a> - Gradio Demo by 🤗 Hugging Face
                </p>
            </div>
           """
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch()