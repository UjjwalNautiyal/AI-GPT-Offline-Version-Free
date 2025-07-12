import tkinter as tk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import threading

# Load the DialoGPT model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
chat_history_ids = None

# GUI window setup
root = tk.Tk()
root.title("AI Chatbot - Offline")
root.geometry("500x620")

# Chat display area
chat_log = tk.Text(root, bg="black", fg="white", font=("Consolas", 12))
chat_log.pack(padx=10, pady=(10, 0), fill=tk.BOTH, expand=True)

# Entry box
entry = tk.Entry(root, font=("Consolas", 14))
entry.pack(padx=10, pady=(10, 5), fill=tk.X)

# Footer label with your name
footer = tk.Label(root, text="Made by Ujjwal Nautiyal", font=("Arial", 10), fg="gray")
footer.pack(pady=(0, 5))

# Function to generate AI response
def respond():
    global chat_history_ids
    user_input = entry.get()
    entry.delete(0, tk.END)

    chat_log.insert(tk.END, f"You: {user_input}\n")

    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = torch.cat([chat_history_ids, inputs], dim=-1) if chat_history_ids is not None else inputs

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8
    )

    reply = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    chat_log.insert(tk.END, f"AI: {reply}\n\n")
    chat_log.see(tk.END)

# Threaded entry handler
def on_enter(event=None):
    threading.Thread(target=respond).start()

entry.bind("<Return>", on_enter)

# Launch the app
root.mainloop()
