"""
Abstracted utility functions for database-agnostic operations.
"""
import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import openai
import re
import time

from database.base import VectorDatabaseProvider, DocumentChunk, CodeExample

# Load OpenAI API key for embeddings
openai.api_key = os.getenv("OPENAI_API_KEY")

def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single API call.
    
    Args:
        texts: List of texts to create embeddings for
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
    
    max_retries = 3
    retry_delay = 1.0  # Start with 1 second delay
    
    for retry in range(max_retries):
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small", # Hardcoding embedding model for now, will change this later to be more dynamic
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            if retry < max_retries - 1:
                print(f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed to create batch embeddings after {max_retries} attempts: {e}")
                # Try creating embeddings one by one as fallback
                print("Attempting to create embeddings individually...")
                embeddings = []
                successful_count = 0
                
                for i, text in enumerate(texts):
                    try:
                        individual_response = openai.embeddings.create(
                            model="text-embedding-3-small",
                            input=[text]
                        )
                        embeddings.append(individual_response.data[0].embedding)
                        successful_count += 1
                    except Exception as individual_error:
                        print(f"Failed to create embedding for text {i}: {individual_error}")
                        # Add zero embedding as fallback
                        embeddings.append([0.0] * 1536)
                
                print(f"Successfully created {successful_count}/{len(texts)} embeddings individually")
                return embeddings

def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using OpenAI's API.
    
    Args:
        text: Text to create an embedding for
        
    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * 1536
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return empty embedding if there's an error
        return [0.0] * 1536

def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.
    
    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    model_choice = os.getenv("MODEL_CHOICE")
    
    try:
        # Create the prompt for generating contextual information
        prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Call the OpenAI API to generate contextual information
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise contextual information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        # Extract the generated context
        context = response.choices[0].message.content.strip()
        
        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"
        
        return contextual_text, True
    
    except Exception as e:
        print(f"Error generating contextual embedding: {e}. Using original chunk instead.")
        return chunk, False

def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.
    
    Args:
        args: Tuple containing (url, content, full_document)
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    url, content, full_document = args
    return generate_contextual_embedding(full_document, content)

async def add_documents_to_database(
    vector_db: VectorDatabaseProvider,
    urls: List[str], 
    chunk_numbers: List[int],
    contents: List[str], 
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20
) -> None:
    """Add documents to the vector database using abstraction layer"""
    
    # Apply contextual embedding if enabled
    use_contextual_embeddings = os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
    
    if use_contextual_embeddings:
        processed_contents = []
        for i, content in enumerate(contents):
            url = urls[i]
            full_document = url_to_full_document.get(url, "")
            contextual_text, success = generate_contextual_embedding(full_document, content)
            processed_contents.append(contextual_text)
            if success:
                metadatas[i]["contextual_embedding"] = True
    else:
        processed_contents = contents
    
    # Create embeddings
    embeddings = create_embeddings_batch(processed_contents)
    
    # Convert to DocumentChunk objects
    documents = []
    for i, content in enumerate(processed_contents):
        parsed_url = urlparse(urls[i])
        source_id = parsed_url.netloc or parsed_url.path
        
        doc = DocumentChunk(
            url=urls[i],
            chunk_number=chunk_numbers[i],
            content=content,
            metadata=metadatas[i],
            embedding=embeddings[i],
            source_id=source_id
        )
        documents.append(doc)
    
    # Add to database
    await vector_db.add_documents(documents, batch_size)

async def search_documents_abstracted(
    vector_db: VectorDatabaseProvider,
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None,
    use_hybrid: bool = False
) -> List[Dict[str, Any]]:
    """Search documents using the abstraction layer"""
    
    query_embedding = create_embedding(query)
    
    if use_hybrid:
        results = await vector_db.hybrid_search_documents(
            query_embedding=query_embedding,
            keyword_query=query,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
    else:
        results = await vector_db.search_documents(
            query_embedding=query_embedding,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
    
    # Convert to dictionary format for JSON serialization
    return [
        {
            "id": result.id,
            "url": result.url,
            "content": result.content,
            "metadata": result.metadata,
            "similarity": result.similarity,
            "source_id": result.source_id
        }
        for result in results
    ]

def extract_code_blocks(markdown_content: str, min_length: int = 1000) -> List[Dict[str, Any]]:
    """
    Extract code blocks from markdown content along with context.
    
    Args:
        markdown_content: The markdown content to extract code blocks from
        min_length: Minimum length of code blocks to extract (default: 1000 characters)
        
    Returns:
        List of dictionaries containing code blocks and their context
    """
    code_blocks = []
    
    # Skip if content starts with triple backticks (edge case for files wrapped in backticks)
    content = markdown_content.strip()
    start_offset = 0
    if content.startswith('```'):
        # Skip the first triple backticks
        start_offset = 3
        print("Skipping initial triple backticks")
    
    # Find all occurrences of triple backticks
    backtick_positions = []
    pos = start_offset
    while True:
        pos = markdown_content.find('```', pos)
        if pos == -1:
            break
        backtick_positions.append(pos)
        pos += 3
    
    # Process pairs of backticks
    i = 0
    while i < len(backtick_positions) - 1:
        start_pos = backtick_positions[i]
        end_pos = backtick_positions[i + 1]
        
        # Extract the content between backticks
        code_section = markdown_content[start_pos+3:end_pos]
        
        # Check if there's a language specifier on the first line
        lines = code_section.split('\n', 1)
        if len(lines) > 1:
            # Check if first line is a language specifier (no spaces, common language names)
            first_line = lines[0].strip()
            if first_line and not ' ' in first_line and len(first_line) < 20:
                language = first_line
                code_content = lines[1].strip() if len(lines) > 1 else ""
            else:
                language = ""
                code_content = code_section.strip()
        else:
            language = ""
            code_content = code_section.strip()
        
        # Skip if code block is too short
        if len(code_content) < min_length:
            i += 2  # Move to next pair
            continue
        
        # Extract context before (1000 chars)
        context_start = max(0, start_pos - 1000)
        context_before = markdown_content[context_start:start_pos].strip()
        
        # Extract context after (1000 chars)
        context_end = min(len(markdown_content), end_pos + 3 + 1000)
        context_after = markdown_content[end_pos + 3:context_end].strip()
        
        code_blocks.append({
            'code': code_content,
            'language': language,
            'context_before': context_before,
            'context_after': context_after,
            'full_context': f"{context_before}\n\n{code_content}\n\n{context_after}"
        })
        
        # Move to next pair (skip the closing backtick we just processed)
        i += 2
    
    return code_blocks

def generate_code_example_summary(code: str, context_before: str, context_after: str) -> str:
    """
    Generate a summary for a code example using its surrounding context.
    
    Args:
        code: The code example
        context_before: Context before the code
        context_after: Context after the code
        
    Returns:
        A summary of what the code example demonstrates
    """
    model_choice = os.getenv("MODEL_CHOICE")
    
    # Create the prompt
    prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example>
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated.
"""
    
    try:
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise code example summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error generating code example summary: {e}")
        return "Code example for demonstration purposes."

async def add_code_examples_to_database(
    vector_db: VectorDatabaseProvider,
    urls: List[str],
    chunk_numbers: List[int],
    code_examples: List[str],
    summaries: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = 20
):
    """Add code examples to the database using abstraction layer"""
    if not urls:
        return
    
    # Create combined texts for embedding (code + summary)
    combined_texts = []
    for i in range(len(code_examples)):
        combined_text = f"{code_examples[i]}\n\nSummary: {summaries[i]}"
        combined_texts.append(combined_text)
    
    # Create embeddings for the batch
    embeddings = create_embeddings_batch(combined_texts)
    
    # Convert to CodeExample objects
    examples = []
    for i, embedding in enumerate(embeddings):
        parsed_url = urlparse(urls[i])
        source_id = parsed_url.netloc or parsed_url.path
        
        example = CodeExample(
            url=urls[i],
            chunk_number=chunk_numbers[i],
            content=code_examples[i],
            summary=summaries[i],
            metadata=metadatas[i],
            embedding=embedding,
            source_id=source_id
        )
        examples.append(example)
    
    # Add to database
    await vector_db.add_code_examples(examples, batch_size)

async def search_code_examples_abstracted(
    vector_db: VectorDatabaseProvider,
    query: str, 
    match_count: int = 10, 
    filter_metadata: Optional[Dict[str, Any]] = None,
    use_hybrid: bool = False
) -> List[Dict[str, Any]]:
    """Search code examples using the abstraction layer"""
    
    # Create a more descriptive query for better embedding match
    enhanced_query = f"Code example for {query}\n\nSummary: Example code showing {query}"
    query_embedding = create_embedding(enhanced_query)
    
    if use_hybrid:
        results = await vector_db.hybrid_search_code_examples(
            query_embedding=query_embedding,
            keyword_query=query,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
    else:
        results = await vector_db.search_code_examples(
            query_embedding=query_embedding,
            match_count=match_count,
            filter_metadata=filter_metadata
        )
    
    # Convert to dictionary format for JSON serialization
    return [
        {
            "id": result.id,
            "url": result.url,
            "content": result.content,
            "metadata": result.metadata,
            "similarity": result.similarity,
            "source_id": result.source_id
        }
        for result in results
    ]

def extract_source_summary(source_id: str, content: str, max_length: int = 500) -> str:
    """
    Extract a summary for a source from its content using an LLM.
    
    This function uses the OpenAI API to generate a concise summary of the source content.
    
    Args:
        source_id: The source ID (domain)
        content: The content to extract a summary from
        max_length: Maximum length of the summary
        
    Returns:
        A summary string
    """
    # Default summary if we can't extract anything meaningful
    default_summary = f"Content from {source_id}"
    
    if not content or len(content.strip()) == 0:
        return default_summary
    
    # Get the model choice from environment variables
    model_choice = os.getenv("MODEL_CHOICE")
    
    # Limit content length to avoid token limits
    truncated_content = content[:25000] if len(content) > 25000 else content
    
    # Create the prompt for generating the summary
    prompt = f"""<source_content>
{truncated_content}
</source_content>

The above content is from the documentation for '{source_id}'. Please provide a concise summary (3-5 sentences) that describes what this library/tool/framework is about. The summary should help understand what the library/tool/framework accomplishes and the purpose.
"""
    
    try:
        # Call the OpenAI API to generate the summary
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise library/tool/framework summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        # Extract the generated summary
        summary = response.choices[0].message.content.strip()
        
        # Ensure the summary is not too long
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
            
        return summary
    
    except Exception as e:
        print(f"Error generating summary with LLM for {source_id}: {e}. Using default summary.")
        return default_summary