"""
LLM Tools
=========

Wrapper for LLM interactions (OpenAI).
Provides simple interfaces for prompting and structured outputs.
"""

import os
import json
from typing import Optional, Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    Simple wrapper for OpenAI LLM interactions.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7
    ):
        """
        Initialize LLM client.
        
        Args:
            api_key: OpenAI API key
            model: Default model to use
            temperature: Default temperature
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.default_model = model
        self.default_temperature = temperature
    
    def prompt(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 2000
    ) -> str:
        """
        Send a simple prompt to the LLM.
        
        Args:
            user_message: User message/prompt
            system_message: Optional system message
            model: Model to use (defaults to default_model)
            temperature: Temperature (defaults to default_temperature)
            max_tokens: Maximum tokens in response
            
        Returns:
            LLM response as string
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": user_message})
        
        response = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def prompt_json(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Prompt the LLM and parse JSON response.
        
        Args:
            user_message: User message (should ask for JSON)
            system_message: Optional system message
            model: Model to use
            temperature: Temperature
            max_tokens: Max tokens
            
        Returns:
            Parsed JSON as dictionary
        """
        # Add JSON instruction to system message
        json_instruction = "\n\nIMPORTANT: You MUST respond with valid JSON only. No other text."
        
        if system_message:
            system_message = system_message + json_instruction
        else:
            system_message = "You are a helpful assistant that responds in JSON format." + json_instruction
        
        response_text = self.prompt(
            user_message=user_message,
            system_message=system_message,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Try to parse JSON
        try:
            # Sometimes the model wraps JSON in markdown code blocks
            if "```json" in response_text:
                # Extract JSON from code block
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON response: {e}")
            print(f"Raw response: {response_text[:500]}...")
            # Return as best-effort dict
            return {"error": "Failed to parse JSON", "raw_response": response_text}
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 2000
    ) -> str:
        """
        Multi-turn chat interface.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
            temperature: Temperature
            max_tokens: Max tokens
            
        Returns:
            LLM response
        """
        response = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature if temperature is not None else self.default_temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def judge_answer_quality(
        self,
        query: str,
        answer: str,
        context: Optional[str] = None,
        reference_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to judge the quality of a RAG answer.
        
        Args:
            query: Original query
            answer: Answer to evaluate
            context: Optional context used
            reference_answer: Optional reference/gold answer
            
        Returns:
            Dictionary with score and reasoning
        """
        system_message = """You are an expert evaluator of question-answering systems.
Your task is to evaluate the quality of answers on a scale of 0-10.

Consider:
- Relevance: Does the answer address the question?
- Accuracy: Is the information correct?
- Completeness: Is the answer complete?
- Clarity: Is the answer well-structured and clear?
- Citation: Does it properly cite sources when provided?

Respond in JSON format with:
{
    "score": <number 0-10>,
    "reasoning": "<brief explanation>",
    "strengths": ["<strength1>", ...],
    "weaknesses": ["<weakness1>", ...]
}"""

        user_message = f"""Evaluate this answer:

Query: {query}

Answer: {answer}
"""
        
        if reference_answer:
            user_message += f"\nReference Answer: {reference_answer}\n"
        
        if context:
            user_message += f"\nContext Provided: {context[:500]}...\n"
        
        user_message += "\nProvide your evaluation in JSON format."
        
        return self.prompt_json(
            user_message=user_message,
            system_message=system_message,
            temperature=0.3  # Lower temperature for more consistent evaluation
        )
    
    def compare_answers(
        self,
        query: str,
        answer_a: str,
        answer_b: str,
        label_a: str = "Answer A",
        label_b: str = "Answer B"
    ) -> Dict[str, Any]:
        """
        Compare two answers and determine which is better.
        
        Args:
            query: Original query
            answer_a: First answer
            answer_b: Second answer
            label_a: Label for first answer
            label_b: Label for second answer
            
        Returns:
            Dictionary with comparison results
        """
        system_message = """You are an expert evaluator comparing two answers to the same question.

Determine which answer is better based on:
- Relevance
- Accuracy
- Completeness
- Clarity

Respond in JSON format:
{
    "winner": "<A or B or tie>",
    "confidence": <0-10>,
    "reasoning": "<explanation>",
    "score_a": <0-10>,
    "score_b": <0-10>
}"""

        user_message = f"""Compare these answers to the question:

Query: {query}

{label_a}: {answer_a}

{label_b}: {answer_b}

Which answer is better? Provide your comparison in JSON format."""

        return self.prompt_json(
            user_message=user_message,
            system_message=system_message,
            temperature=0.3
        )


# Convenience function for quick LLM calls
def quick_prompt(message: str, system: Optional[str] = None) -> str:
    """
    Quick one-off LLM prompt.
    
    Args:
        message: User message
        system: Optional system message
        
    Returns:
        LLM response
    """
    client = LLMClient()
    return client.prompt(message, system_message=system)


if __name__ == "__main__":
    # Test LLM tools
    print("=== Testing LLM Tools ===\n")
    
    client = LLMClient()
    
    # Test simple prompt
    response = client.prompt("What is RAG in the context of AI?")
    print("Simple prompt response:")
    print(response[:200] + "...\n")
    
    # Test JSON prompt
    json_response = client.prompt_json(
        "List 3 key components of a RAG system in JSON format with keys: components (array of strings)"
    )
    print("JSON response:")
    print(json_response)
    print()
    
    # Test answer judging
    judgment = client.judge_answer_quality(
        query="What is GDPR?",
        answer="GDPR is the General Data Protection Regulation, a European law about data privacy."
    )
    print("Answer quality judgment:")
    print(f"Score: {judgment.get('score', 'N/A')}")
    print(f"Reasoning: {judgment.get('reasoning', 'N/A')}")

