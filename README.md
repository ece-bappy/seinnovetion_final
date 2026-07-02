# AI Assignment Grader

This app grades student submissions against a course book using a retrieval-augmented workflow with Streamlit, OpenAI embeddings, and an LLM-based grading pipeline.

## What the app does

- Upload a questions file or paste questions directly.
- Save the questions with the dedicated button.
- Upload one student answer file for single evaluation.
- Upload multiple student answer files for bulk evaluation.
- Retrieve book context relevant to each rubric criterion and grade the answer against that context.
- Show a structured grading report with criterion-level feedback and flags.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file with your OpenAI credentials:
   ```env
   OPENAI_API_KEY=your-key-here
   OPENAI_MODEL=gpt-5.4-nano
   OPENAI_EMBEDDING_MODEL=text-embedding-3-small
   ```
3. Place the course book files in the `book/` folder.

## Run

```bash
streamlit run app.py
```

## How to use it

1. Open the app in your browser.
2. In the Questions section, upload a PDF/TXT/MD containing the assignment questions or paste the questions manually.
3. Click Save questions.
4. In Single grading, upload one student answer file and click Evaluate.
5. In Bulk grading, upload multiple student answer files and click Evaluate all.
6. Create a folder named ```/book``` and put the book there

## Notes

- If no OpenAI API key is available, the app will fall back to a lightweight heuristic mode so the interface can still be tested.
- The grading is designed to be grounded in the retrieved book context rather than relying only on the student answer text.
