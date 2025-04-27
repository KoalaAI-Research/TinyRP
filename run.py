#!/usr/bin/env python3
# model_inference.py - Simple application to interact with a trained ChatML model
import os
import torch
import argparse
from transformers import LlamaForCausalLM, AutoTokenizer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
import time

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Interact with a trained ChatML model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="./chatml-tiny-model",
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512,
        help="Maximum length of generated text"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature (higher = more creative, lower = more focused)"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9,
        help="Nucleus sampling parameter (lower = more deterministic)"
    )
    return parser.parse_args()

def get_device():
    """Determine the appropriate device for running the model."""
    if torch.cuda.is_available():
        device = "cuda"
        console.print(f"[green]Using NVIDIA GPU: {torch.cuda.get_device_name(0)}[/green]")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = "xpu"
        console.print("[green]Using Intel XPU (Arc)[/green]")
    else:
        device = "cpu"
        console.print("[yellow]No GPU detected, using CPU (this will be slow)[/yellow]")
    return device

def load_model(model_path, device):
    """Load the model and tokenizer from the specified path."""
    try:
        console.print(f"Loading model from [bold]{model_path}[/bold]...")
        # First try loading the tokenizer from the model path
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            console.print(f"[green]Loaded tokenizer from model path[/green]")
        except Exception as e:
            console.print(f"[yellow]Error loading tokenizer from model path: {str(e)}[/yellow]")
            console.print("Attempting to load tokenizer from original location...")
            tokenizer = AutoTokenizer.from_pretrained("chatml_tokenizer")
            
        # Ensure special tokens are set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load the model
        model = LlamaForCausalLM.from_pretrained(model_path)
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        console.print(f"[green]Model loaded successfully![/green]")
        console.print(f"Model has [bold]{sum(p.numel() for p in model.parameters()):,}[/bold] parameters")
        
        return model, tokenizer
    except Exception as e:
        console.print(f"[red]Error loading model: {str(e)}[/red]")
        return None, None

def format_chatml_prompt(user_input):
    """Format user input as a ChatML prompt."""
    return f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

def generate_response(model, tokenizer, prompt, device, max_length=512, temperature=0.7, top_p=0.9):
    """Generate a response from the model."""
    try:
        # Format the prompt for ChatML
        chatml_prompt = format_chatml_prompt(prompt)
        
        # Tokenize the prompt
        inputs = tokenizer(chatml_prompt, return_tensors="pt").to(device)
        
        # Track generation time
        start_time = time.time()
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode the response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract just the assistant's response
        try:
            assistant_response = full_response.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0].strip()
        except:
            # If the model output doesn't follow the expected format, return the whole thing minus the prompt
            assistant_response = full_response[len(chatml_prompt):].strip()
        
        # Calculate generation time and tokens per second
        end_time = time.time()
        generation_time = end_time - start_time
        num_new_tokens = len(outputs[0]) - len(inputs.input_ids[0])
        tokens_per_second = num_new_tokens / generation_time if generation_time > 0 else 0
        
        console.print(f"\n[dim]Generated {num_new_tokens} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/s)[/dim]")
        
        return assistant_response
    except Exception as e:
        console.print(f"[red]Error generating response: {str(e)}[/red]")
        return "Sorry, I encountered an error while generating a response."

def main():
    """Main application function."""
    args = parse_args()
    
    # Print header
    console.print(Panel.fit("ChatML Tiny Model Interface", style="bold blue"))
    
    # Get device
    device = get_device()
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path, device)
    if model is None or tokenizer is None:
        console.print("[red]Failed to load model. Exiting...[/red]")
        return
    
    console.print(Panel.fit(
        "Type your message and press Enter to get a response.\n"
        "Type 'exit', 'quit', or 'q' to exit.\n"
        "Type '/params' to view or change generation parameters.",
        title="Instructions",
        style="green"
    ))
    
    # Set initial parameters
    params = {
        "max_length": args.max_length,
        "temperature": args.temperature,
        "top_p": args.top_p
    }
    
    # Main interaction loop
    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
            
            # Check for exit commands
            if user_input.lower() in ('exit', 'quit', 'q'):
                console.print("[yellow]Exiting...[/yellow]")
                break
                
            # Check for parameter commands
            if user_input.lower() == '/params':
                console.print(Panel.fit(
                    f"Current Parameters:\n"
                    f"max_length: {params['max_length']}\n"
                    f"temperature: {params['temperature']}\n"
                    f"top_p: {params['top_p']}",
                    title="Generation Parameters",
                    style="cyan"
                ))
                
                # Allow parameter changes
                change = Prompt.ask("Change parameters? [y/N]")
                if change.lower() == 'y':
                    params['max_length'] = int(Prompt.ask("max_length", default=str(params['max_length'])))
                    params['temperature'] = float(Prompt.ask("temperature", default=str(params['temperature'])))
                    params['top_p'] = float(Prompt.ask("top_p", default=str(params['top_p'])))
                    console.print("[green]Parameters updated![/green]")
                continue
                
            # Generate and display response
            with console.status("[bold green]Generating response...[/bold green]"):
                response = generate_response(
                    model, 
                    tokenizer, 
                    user_input, 
                    device,
                    max_length=params['max_length'],
                    temperature=params['temperature'],
                    top_p=params['top_p']
                )
            
            console.print("\n[bold purple]Assistant[/bold purple]")
            console.print(Panel(response, style="purple"))
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user. Exiting...[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            
if __name__ == "__main__":
    main()
