import os
from uuid import uuid4
from dotenv import load_dotenv
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.eleven_labs import ElevenLabsTools
from agno.tools.firecrawl import FirecrawlTools
from agno.utils.audio import write_audio_to_file
from agno.utils.log import logger
import streamlit as st

# Load biến môi trường từ file .env
load_dotenv()

# Override FirecrawlTools để fix lỗi JSON serialization
class FixedFirecrawlTools(FirecrawlTools):
    def scrape_website(self, url: str) -> str:
        result = self.app.scrape_url(url)
        data = result.model_dump()
        content = data.get('markdown') or data.get('html') or ''
        if not isinstance(content, str):
            content = str(content)
        return content

# Load API keys từ env
openai_api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")

# Kiểm tra key
keys_provided = all([openai_api_key, elevenlabs_api_key, firecrawl_api_key])

# Set env để Agno và các tools dùng
if keys_provided:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["ELEVEN_LABS_API_KEY"] = elevenlabs_api_key
    os.environ["FIRECRAWL_API_KEY"] = firecrawl_api_key

# Streamlit Page Setup
st.set_page_config(page_title="📰 ➡️ 🎙️ Blog to Podcast Agent", page_icon="🎙️")
st.title("📰 ➡️ 🎙️ Blog to Podcast Agent")

if not keys_provided:
    st.error("API keys are not set in the environment variables. Please add them to your .env file.")

url = st.text_input("Enter the Blog URL:", "https://vneconomy.vn/thu-tuong-chu-tri-hoi-nghi-truc-tuyen-voi-3-noi-dung-quan-trong.htm")
generate_button = st.button("🎙️ Generate Podcast", disabled=not keys_provided or url.strip() == "")

if generate_button:
    with st.spinner("Processing... Scraping blog, summarizing and generating podcast 🎶"):
        try:
            blog_to_podcast_agent = Agent(
                name="Blog to Podcast Agent",
                agent_id="blog_to_podcast_agent",
                model=OpenAIChat(id="gpt-4o"),
                tools=[
                    ElevenLabsTools(
                        voice_id="LPldyaIkUUSOPCRFrgYJ",  # Voice tiếng Việt
                        model_id="eleven_multilingual_v2",
                        target_directory="audio_generations",
                    ),
                    FixedFirecrawlTools(),
                ],
                description="You are an AI agent that can generate audio using the ElevenLabs API.",
                instructions=[
                    "When the user provides a blog URL:",
                    "1. Use FirecrawlTools to scrape the blog content",
                    "2. Create a concise summary of the blog content that is NO MORE than 450 characters long",
                    "3. The summary should capture the main points while being engaging and conversational",
                    "4. Use the ElevenLabsTools to convert the summary to audio",
                    "Ensure the summary is within the 450 character limit to avoid ElevenLabs API limits",
                ],
                markdown=True,
                debug_mode=True,
            )

            # Chạy agent
            podcast: RunResponse = blog_to_podcast_agent.run(f"Convert the blog content to a podcast: {url}")

            save_dir = "audio_generations"
            os.makedirs(save_dir, exist_ok=True)

            if podcast.audio and len(podcast.audio) > 0:
                filename = f"{save_dir}/podcast_{uuid4()}.wav"
                write_audio_to_file(
                    audio=podcast.audio[0].base64_audio,
                    filename=filename
                )

                st.success("Podcast generated successfully! 🎧")
                audio_bytes = open(filename, "rb").read()
                st.audio(audio_bytes, format="audio/wav")

                st.download_button(
                    label="Download Podcast",
                    data=audio_bytes,
                    file_name="generated_podcast.wav",
                    mime="audio/wav"
                )
            else:
                st.error("No audio was generated. Please try again.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.error(f"Streamlit app error: {e}")