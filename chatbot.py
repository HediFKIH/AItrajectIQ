import llminsight
from openai import AzureOpenAI, OpenAI
import os
import random
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load .env file
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
# SDK env vars (see sdk/README.md)
llminsight_api_key = os.getenv("API_KEY")
llminsight_project_id = os.getenv("LLMINSIGHT_PROJECT_ID") 
llminsight_base_url = os.getenv("LLMINSIGHT_BASE_URL") 
#api_key = os.getenv("API_KEY")
#azure_endpoint = os.getenv("AZURE_ENDPOINT")
#deployment_name = os.getenv("DEPLOYMENT_NAME")
#api_version = os.getenv("API_VERSION")

if not llminsight_api_key:
    raise RuntimeError("LLMINSIGHT_API_KEY is not set. Add it to your .env before running the chatbot.")
llminsight.init(
    project_id=llminsight_project_id,
    api_key=llminsight_api_key,
    base_url=llminsight_base_url,
)
llminsight.patch()

# Initialize client with Gemini API key
client = OpenAI(
    api_key=gemini_api_key,
    #api_key="string",
    #api_key=api_key,
    #azure_endpoint=azure_endpoint,
    #api_version=api_version
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Store conversation history
# messages = [
#     {"role": "system", "content": "You are a helpful AI assistant."}
# ]

print("🤖 Gemini Chatbot Started (type 'exit' to quit)\n")

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

documents = [
    "The universe began with the Big Bang approximately 13.8 billion years ago.",
    "A galaxy is a massive system of stars, stellar remnants, interstellar gas, dust, and dark matter.",
    "Black holes are regions in space where the gravitational pull is so strong that nothing, not even light, can escape.",
    "The Milky Way is the galaxy that contains our solar system.",
    "The Sun is a star at the center of our solar system, providing light and heat essential for life on Earth.",
    "Astronomical units (AU) are used to measure distances within our solar system, equal to the average distance from Earth to the Sun.",
    "The Hubble Space Telescope has revolutionized our understanding of the universe by providing high-resolution images of distant objects.",
    "A supernova is an astronomical event that occurs during the last stages of a massive star's life, resulting in a catastrophic explosion.",
    "The moon's surface is covered with craters from impacts with asteroids and comets.",
    "Mars is known as the Red Planet due to its reddish appearance, caused by iron oxide (rust) on its surface.",
    "Saturn's rings are made up of countless particles of ice and rock that orbit the planet.",
    "Jupiter is the largest planet in our solar system and is known for its Great Red Spot, a massive storm system.",
    "The Andromeda Galaxy is the nearest large galaxy to the Milky Way and is on a collision course with our galaxy.",
    "Dark matter is an unknown form of matter that does not emit or interact with electromagnetic radiation, yet its presence is inferred from gravitational effects on visible matter.",
    "A neutron star is a remnant of a massive star that has exploded in a supernova, and it is incredibly dense, with a mass greater than the Sun in a volume smaller than a city.",
    "Comets are icy bodies that originate from the outer regions of the solar system, and their bright tails are caused by the Sun's radiation.",
    "The speed of light is approximately 299,792 kilometers per second, and it serves as the universal speed limit for information transfer.",
    "Exoplanets are planets that orbit stars outside our solar system, and thousands have been discovered using various techniques.",
    "The event horizon is the boundary around a black hole beyond which nothing, not even light, can escape.",
    "The observable universe is approximately 93 billion light-years in diameter.",
    "A solar eclipse occurs when the moon passes between the Earth and the Sun, blocking out the Sun's light.",
    "The Pleiades is a star cluster in the constellation Taurus, also known as the Seven Sisters.",
    "The asteroid belt is a region of space between Mars and Jupiter where the majority of asteroids in our solar system are located.",
    "The Orion Nebula is a massive star-forming region located in the constellation Orion.",
    "The light-year is a unit of distance used in astronomy, equal to the distance that light travels in one year.",
    "A red giant is a type of star that has expanded and cooled after exhausting the hydrogen in its core.",
    "The Kepler Space Telescope was launched to search for exoplanets and discovered thousands of new worlds.",
    "Cosmic background radiation is the faint glow left over from the Big Bang and is an important clue about the early universe.",
    "The Voyager probes were launched in 1977 to explore the outer planets of the solar system and are now in interstellar space.",
    "Neutrinos are subatomic particles with extremely small mass and no electric charge that are produced in large quantities by stars, including the Sun.",
    "The Kuiper Belt is a region of the solar system beyond Neptune that contains many icy bodies, including dwarf planets like Pluto.",
    "The constellation Ursa Major is home to the Big Dipper, one of the most recognizable star patterns in the night sky.",
    "A pulsar is a rotating neutron star that emits beams of electromagnetic radiation, which can be detected as periodic pulses.",
    "The Lagrange points are positions in space where the gravitational forces of a planet and a moon or the Sun and a planet balance out, providing stable locations for objects.",
    "The Crab Nebula is a supernova remnant and pulsar wind nebula in the constellation Taurus.",
    "Astronomers use radio telescopes to detect electromagnetic radiation in the radio frequency range, helping to study distant objects.",
    "Galaxies can be classified into different types based on their shape, including spiral, elliptical, and irregular galaxies.",
    "The constellations of the zodiac are groups of stars that the Sun appears to pass through over the course of a year.",
    "The Great Wall is a massive galactic structure composed of galaxies and is one of the largest known structures in the universe.",
    "The Sun's core is where nuclear fusion takes place, converting hydrogen into helium and releasing vast amounts of energy.",
    "The speed of a spacecraft can be measured using the Doppler effect, by analyzing changes in the frequency of radio waves reflected from the spacecraft.",
    "The magnetic fields of planets, including Earth, are generated by the motion of molten iron in their cores.",
    "The Perseus Cluster is a massive galaxy cluster in the constellation Perseus and contains one of the largest known black holes.",
    "The Solar System consists of the Sun, eight planets, their moons, asteroids, comets, and other objects.",
    "Cosmology is the study of the large-scale properties of the universe as a whole.",
    "Astronomical spectroscopy is a technique used to analyze the light from distant stars and galaxies to determine their composition, temperature, velocity, and other properties.",
    "The asteroid Ceres is the largest object in the asteroid belt and is classified as a dwarf planet.",
    "Meteor showers occur when Earth passes through the debris left behind by a comet, causing meteors to burn up in the atmosphere.",
    "The heliosphere is the bubble of charged particles created by the Sun's solar wind that extends well beyond Pluto.",
    "The James Webb Space Telescope is set to launch in 2021 and will be able to observe distant galaxies and exoplanets in unprecedented detail."
]

doc_embeddings = model.encode(documents)
dim = doc_embeddings.shape[1]

# Cosine similarity via inner product on normalized vectors
index = faiss.IndexFlatL2(dim)
index.add(np.array(doc_embeddings, dtype=np.float32))  # type: ignore

def retrieve(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)  # type: ignore
    return [documents[i] for i in indices[0]]

def generate_response(retrieved_docs, query):
    context_block = "\n".join(f"- {doc}" for doc in retrieved_docs)
    prompt = f"Query: {query}\n\nRelevant context:\n{context_block}"

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use the provided context when relevant and keep your answers brief and to the point."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content


def generate_hallucinated_response(retrieved_docs, query):
    """Generate an intentionally wrong response to test hallucination metrics."""
    context_block = "\n".join(f"- {doc}" for doc in retrieved_docs)
    prompt = f"Query: {query}\n\nRelevant context:\n{context_block}"

    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": (
                "You are a test assistant that MUST generate intentionally incorrect and misleading answers. "
                "Always respond with confident but factually wrong information. "
                "Invert facts, use wrong numbers, swap names, and contradict well-known science. "
                "Never give the correct answer. Keep your answers brief."
            )},
            {"role": "user", "content": prompt}
        ],
        temperature=1.5,
    )
    return response.choices[0].message.content


messages = [
    {"role": "system", "content": "You are a helpful AI assistant."}
]

# Intentionally wrong statements for hallucination metric testing
HALLUCINATION_PHRASES = [
    "The sky is red during a normal clear sunny day.",
    "Water boils at 50°C at sea level.",
    "The Earth has two moons.",
    "Humans can breathe comfortably on Mars without any equipment.",
    "The Great Wall of China is visible from the Moon with the naked eye.",
    "The Sun orbits around the Earth once every 24 hours.",
    "Light travels slower than sound in a vacuum.",
    "Jupiter is the smallest planet in our solar system.",
]

hallucination_mode = True  # toggle for hallucination testing (enabled by default)

while True:
    user_input = input("You: ")

    # Toggle hallucination mode on/off
    if user_input.lower() == "enable_hallucination":
        hallucination_mode = True
        print("[Hallucination mode ENABLED] Any input will return a wrong answer.")
        continue
    if user_input.lower() == "disable_hallucination":
        hallucination_mode = False
        print("[Hallucination mode DISABLED] Normal responses resumed.")
        continue

    # In hallucination mode, randomly return wrong or correct answer
    if hallucination_mode:
        contexts = retrieve(user_input, k=3)
        if random.random() < 0.5:  # 50% chance of hallucination
            #answer = random.choice(HALLUCINATION_PHRASES)
            answer = generate_hallucinated_response(contexts, query=user_input)
            print("Bot (hallucinated):", answer)
        else:
            answer = generate_response(contexts, query=user_input)
            print("Bot (correct):", answer)
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": answer})
        continue
    
    
    if user_input.lower() == "exit":
        print("Chat ended.")
        break

    contexts = retrieve(user_input, k=3)
    assistant_reply = generate_response(contexts, query=user_input)

    # Print reply
    print("Bot:", assistant_reply)

    # Add assistant reply to history
    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": assistant_reply})