# CharMind LLM: Your Personal AI Framework

**A self-contained and easy-to-use framework for training, fine-tuning, and aligning your own private language models.**

This project is designed to be run locally. It provides a simple web interface to manage the entire lifecycle of your models, from data preparation to interactive chat.

---

## Core Features

- **All-in-One UI:** A single web interface to:
    - Chat with your trained models.
    - Select which model to use (pre-trained, fine-tuned, or RLHF-aligned).
    - Launch training, fine-tuning, and RLHF jobs.
    - Monitor the status and logs of training jobs in real-time.
    - Collect preference data for RLHF.
- **Simplified Data Pipeline:** A single script prepares all your data for any kind of training.
- **Organized Checkpoints:** Models from different training stages are saved separately, so you never lose a valuable checkpoint.
- **From-Scratch Implementation:** Core components are built from the ground up using PyTorch for maximum transparency and control.

---

## How to Use

The entire process is designed to be simple and managed from the UI.

### Step 1: Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # The app also uses pandas for data processing
    pip install pandas
    ```

2.  **Add Your Data (Optional):**
    - Place your raw text data for training into `.json` files inside `data/finetune/` or `data/feedback_data/`. The structure should be simple, e.g., `[{"text": "my first sentence"}, {"text": "my second sentence"}]`.
    - The project comes with some sample data to get you started.

### Step 2: Prepare the Data

Run the data preparation script once. This will read all the raw data you've added, process it, and create clean data files for the training scripts.

```bash
python data/prepare_data.py
```
This will consolidate everything into `data/corpus/processed_training_corpus.jsonl` and `data/feedback_data/processed_preference_data.jsonl`.

### Step 3: Launch the Application

Start the web server:
```bash
python app.py
```
Open your browser to **`http://localhost:7860`**.

---

## Using the Web Interface

### Chatting with the Model
- The main screen is the chat interface.
- Use the **Active Model** dropdown in the sidebar to switch between any of your saved models.

### Training a Model

1.  Find the **Training Management** section in the sidebar.
2.  **Select a Training Mode**:
    - `Pre-train`: Train a model from scratch.
    - `Fine-tune`: Further train an existing model using LoRA for efficiency.
    - `RLHF`: Align a model using feedback.
3.  **Enter a Run Name:** Give your training job a unique name (e.g., `my-first-pretrain`).
4.  **(For RLHF only)**: Select a **Base Model** from the dropdown that appears. This should be a model you have already pre-trained or fine-tuned.
5.  Click **Start Training**.

### Monitoring Training
- The **Training Status** section on the right will automatically update.
- You can see which jobs are running, their status (Running, Completed, Failed), and the last few lines of their log output.

### Providing Human Feedback (RLHF)
1.  In the sidebar, click the **RLHF Labeling** button.
2.  In the text input at the bottom, enter a prompt and press Enter.
3.  The model will generate two different responses.
4.  Click on the response you think is better. Your preference will be automatically saved for the next RLHF training run.
5.  Click "Next Prompt" to continue labeling.
