def predict_answer(question, model, tokenizer, max_length=256, num_beams=4):
    # Format the input similar to training
    input_text = f"Answer this financial question based on Warren Buffett's principles: {question}"
    
    # Tokenize the input
    input_tokens = tokenizer(input_text, return_tensors="tf", max_length=256, padding='max_length', truncation=True)
    
    # Generate the answer
    generated_tokens = model.generate(
        input_tokens["input_ids"],
        attention_mask=input_tokens["attention_mask"],
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True
    )
    
    # Decode the generated tokens back to text
    predicted_answer = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    
    return predicted_answer