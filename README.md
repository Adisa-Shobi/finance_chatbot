# Warren Buffett Financial Chatbot

A fine-tuned FLAN-T5 model that answers financial questions based on Warren Buffett's investment principles, trained on 26 years of Berkshire Hathaway shareholder letters (1998-2024).

---

## Features
- Domain-aware: Answers financial questions, politely rejects off-topic queries
- Buffett-style: Value investing and business wisdom
- FastAPI backend: Ready-to-use API

---

## Dataset
- **Source**: [eagle0504/warren-buffett-letters-qna-r1-enhanced-1998-2024](https://huggingface.co/datasets/eagle0504/warren-buffett-letters-qna-r1-enhanced-1998-2024)
- **Structure**: `question`, `answer`, `reasoning`
- **Augmentation**: 75+ out-of-domain questions from different fields (health, ariculture, weather, etc) in `out_of_domain_training_data.csv` for robust domain rejection

This dataset uses letters written from Warren Buffett to his shareholders to derive a question and answer dataset. It contains nuggets of the financial guru's vast experience operating in various markets. Warren Buffett's success lives through Berkshire Hathaway, his investment firm, which he has recently announced he will be stepping down as CEO. Training based on this dataset aims to preserve his financial savvy using ML/AI techniques by responding based on his philosophy. The dataset spans 26 years (1998-2024) of proven investment wisdom, structured as Q&A pairs with detailed reasoning that reflects his value investing principles. To ensure responsible AI behavior, the dataset is augmented with 75+ out-of-domain questions from fields like health, agriculture, and weather, teaching the model to maintain appropriate boundaries and politely redirect non-financial queries to relevant investment topics.

---

## Performance Metrics

### BLEU
This metric tracks the number of words/phrases that appear in both the genereted text and the original text.
It is used to determine if the model is using the right terminology in it's responses. It focuses more on precision.

### ROUGE-1 (0-1, higher better)
Counts individual word overlap between generated and reference text. Measures content coverage - did the model capture the key concepts? Best indicator of whether important information is included.

### ROUGE-2 (0-1, higher better)
Measures 2-word sequence overlap, stricter than ROUGE-1. Evaluates fluency and word order preservation. Lower scores indicate poor sentence structure.

### ROUGE-L (0-1, higher better)
Finds longest common word sequence, allowing gaps between words. Measures overall similarity with flexible matching. Good for detecting if responses follow similar logical flow.

### Perplexity (lower better, 1-∞)
Measures how "surprised" the model is by the reference text. Lower values mean better language modeling and prediction confidence. Values above 50 indicate model confusion.



---

## Quick Start

### 1. Local (Python)
```bash
pip install -r requirements.txt
python app.py
```
API: `POST /predict` at `http://localhost:8000`

#### Example
```python
import requests
resp = requests.post("http://localhost:8000/predict", json={"question": "What do you consider to be a valuable company?"})
print(resp.json())
```

---

### 2. Docker

#### Build & Run with Docker Compose
```bash
docker compose up
```
API will be available at `http://localhost:8000`

**NOTE**: You need to have docker installed on your machine

---

## Project Structure
```
finance_chatbot/
├── app.py                # FastAPI app
├── src/                  # Backend code
├── pretrained/           # Fine-tuned model
├── out_of_domain_training_data.csv
├── requirements.txt
├── DockerFile
├── docker-compose.yml
```

---

## Example Chats

[Web app](https://warren-wisdom-chat.vercel.app/)

### Chatbot Welcome Page
![Chatbot](https://i.imgur.com/76RRutv.png)

### Demo answering investment questions
![Chatbot](https://i.imgur.com/a2K3lAi.png)

### Rejecting question outside finance domain
![Chatbot](https://i.imgur.com/Rob4b5w.png)



