# LectureLens Evaluation

## Configuration

- Date: 2026/11/13
- Python version: 3.17
- Operating system: Windows 10pro
- CPU / RAM / GPU: 
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Ollama model:llama3.2:3b
- Chunk size: 180 tokens
- Overlap: 30 tokens
- Top-k: 2
- Minimum similarity: 0.30
- Documents and page counts: 8

## Development Results

Use these questions to adjust retrieval settings.

| ID | Question | Expected answer/pages | Retrieved relevant page? | Answer supported? | Correct abstention? | Seconds |
|S1|Definition of the word 'Hardware'.|provides basic computing resources 
(CPU, memory, I/O devices).|Yes|Yes|No|10.2|

## Final Evaluation

Use different questions after freezing the settings.

| ID | Question | Expected answer/pages | Retrieved relevant page? | Answer supported? | Correct abstention? | Seconds |
|S1|Without an OS what will happen?| No multitasking.
 No user interface.
 No resource management.
 No security controls./ one page
|Yes|Yes|No|73.6|

## Summary

- Retrieval hit rate: 75%
- Fully supported answer rate: 75%
- Correct abstention rate: N/A
- Median response time: 41.9 sec

## Observed Failures

Describe unsuccessful examples and possible causes.