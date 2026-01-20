"""
LLM-Based Entity Extraction - Base Class
========================================

This module extracts structured entities from unstructured text using Large
Language Models. It serves as the base class for domain-specific extractors.

What It Does
------------
- Configures LLM providers (OpenAI GPT-4o or Google Gemini)
- Sends text with prompts and few-shot examples to the LLM
- Parses responses into structured extractions with entity class and text
- Tracks character positions of extracted entities in the source text
- Saves results to JSONL files or generates HTML visualizations

Supported LLM Providers
-----------------------
- GEMINI: Google's Gemini 2.5 Pro model (requires LANGEXTRACT_API_KEY)
- OPENAI: OpenAI's GPT-4o model (requires OPENAI_API_KEY)

Output Formats
--------------
- In-memory: langextract result objects with extractions
- JSONL: Annotated documents for further processing
- HTML: Interactive visualization for review

Example Usage
-------------
    from extract_graph.extractors.langextractor import LangExtractor
    import langextract as lx

    # Create extractor with OpenAI
    extractor = LangExtractor("OPENAI")

    # Configure extraction
    extractor.setPrompt("Extract person names and their roles.")
    extractor.setInputText("Alice is the CEO. Bob is the CTO.")
    extractor.setExamples([
        lx.data.ExampleData(
            text="John is the manager.",
            extractions=[
                lx.data.Extraction(extraction_class="person", extraction_text="John"),
                lx.data.Extraction(extraction_class="role", extraction_text="manager"),
            ]
        )
    ])

    # Execute extraction
    result = extractor.extract()

    # View results
    extractor.displayEntitiesWithPosition()
    extractor.saveResults("extraction_output")

See Also
--------
- extract_graph.extractors.extracttechnologies : Specialized work history extractor
- langextract documentation : https://github.com/langextract/langextract
"""

from dotenv import load_dotenv
import langextract as lx
import os
import textwrap
import logging

# Load environment variables for API keys.
load_dotenv()


class LangExtractor:
    """
    Base class for LLM-based entity extraction from text.

    This class provides a flexible framework for extracting structured
    entities from unstructured text using large language models. It
    supports multiple LLM providers and outputs results in various formats.

    The class implements the Template Method pattern: the extraction
    workflow is defined here, while subclasses customize prompts and
    examples for domain-specific extraction tasks.

    Attributes
    ----------
    model : str
        The LLM provider being used ('GEMINI' or 'OPENAI').
    prompt : str
        Instructions for the LLM describing what to extract.
    input_text : str
        The text to extract entities from.
    examples : List[lx.data.ExampleData]
        Few-shot examples guiding the extraction.
    result : lx.data.AnnotatedDocument
        The extraction results (populated after extract() is called).
    logger : logging.Logger
        Logger instance for this class.

    Parameters
    ----------
    model : str
        LLM provider to use. Must be 'GEMINI' or 'OPENAI'.

    Raises
    ------
    ValueError
        If an unsupported model is specified.

    Examples
    --------
    Basic extraction setup:

        >>> extractor = LangExtractor("OPENAI")
        >>> extractor.setPrompt("Extract company names mentioned.")
        >>> extractor.setInputText("I worked at Google and Microsoft.")
        >>> extractor.setExamples([...])  # Provide few-shot examples
        >>> result = extractor.extract()

    Notes
    -----
    Environment Variables Required:
        - OPENAI_API_KEY: For OpenAI models
        - LANGEXTRACT_API_KEY: For Gemini models

    The langextract library handles:
        - LLM API communication
        - Prompt construction with examples
        - Response parsing into structured extractions
        - Character position tracking for extractions
    """

    def __init__(self, model: str):
        """
        Initialize the extractor with the specified LLM provider.

        Parameters
        ----------
        model : str
            LLM provider to use. Must be 'GEMINI' or 'OPENAI'.

        Raises
        ------
        ValueError
            If model is not 'GEMINI' or 'OPENAI', or if the required
            API key environment variable is not set.

        Notes
        -----
        API keys are validated during initialization following the
        fail-fast principle. This ensures configuration errors are
        caught early rather than during extraction.
        """
        self.logger = logging.getLogger(__name__)

        # Validate and configure the LLM provider.
        # Each provider requires a specific API key environment variable.
        # We fail fast if the API key is missing (LSP compliance - __init__
        # should either succeed completely or raise an exception).
        match model:
            case "GEMINI":
                self.logger.info("Creating a GEMINI instance")
                key = os.getenv("LANGEXTRACT_API_KEY")
                if not key:
                    raise ValueError(
                        "LANGEXTRACT_API_KEY environment variable is not set. "
                        "Required for GEMINI model."
                    )
                self.model = model

            case "OPENAI":
                self.logger.info("Creating an OPENAI instance")
                key = os.getenv("OPENAI_API_KEY")
                if not key:
                    raise ValueError(
                        "OPENAI_API_KEY environment variable is not set. "
                        "Required for OPENAI model."
                    )
                self.model = model

            case _:
                raise ValueError(f"model {model} must be one of `GEMINI` or `OPENAI`")

    def setPrompt(self, prompt: str) -> None:
        """
        Set the extraction instructions for the LLM.

        The prompt describes what entities to extract from the text.
        It should be clear and specific about the entity types and
        any constraints on extraction.

        Parameters
        ----------
        prompt : str
            Extraction instructions. Will be dedented to remove
            leading whitespace from multi-line strings.

        Examples
        --------
            >>> extractor.setPrompt('''
            ...     Extract the following from job descriptions:
            ...     - Company names
            ...     - Job titles
            ...     - Required technologies
            ... ''')
        """
        self.prompt = textwrap.dedent(prompt)

    def setInputText(self, inputText: str) -> None:
        """
        Set the text to extract entities from.

        Parameters
        ----------
        inputText : str
            The source text for entity extraction.
            Can be any length, but very long texts may be
            truncated by the LLM's context window.

        Examples
        --------
            >>> extractor.setInputText(interview_transcript)
        """
        self.input_text = inputText

    def setExamples(self, examples: list) -> None:
        """
        Set few-shot examples to guide the extraction.

        Examples demonstrate the expected extraction behavior to the LLM.
        Good examples significantly improve extraction accuracy.

        Parameters
        ----------
        examples : List[lx.data.ExampleData]
            List of example extractions. Each example contains:
            - text: Sample input text
            - extractions: Expected extraction results

        Examples
        --------
            >>> extractor.setExamples([
            ...     lx.data.ExampleData(
            ...         text="John works at Acme Corp as a developer.",
            ...         extractions=[
            ...             lx.data.Extraction(
            ...                 extraction_class="company",
            ...                 extraction_text="Acme Corp"
            ...             ),
            ...             lx.data.Extraction(
            ...                 extraction_class="job_title",
            ...                 extraction_text="developer"
            ...             ),
            ...         ]
            ...     )
            ... ])

        Notes
        -----
        Best practices for examples:
            - Provide 2-5 diverse examples
            - Cover edge cases and variations
            - Match the domain of your input text
            - Be consistent in extraction style
        """
        self.examples = examples

    def extract(self) -> object:
        """
        Execute the entity extraction using the configured LLM.

        Sends the prompt, input text, and examples to the LLM and
        parses the response into structured extractions.

        Returns
        -------
        lx.data.AnnotatedDocument
            Extraction results containing:
            - extractions: List of extracted entities
            - Each extraction has: extraction_class, extraction_text, char_interval

        Raises
        ------
        ValueError
            If prompt, input_text, or examples are not set.
        openai.error.AuthenticationError
            If OpenAI API key is invalid.
        Exception
            If LLM API call fails.

        Examples
        --------
            >>> result = extractor.extract()
            >>> for entity in result.extractions:
            ...     print(f"{entity.extraction_class}: {entity.extraction_text}")

        Notes
        -----
        Model-specific behavior:
            - GEMINI: Uses gemini-2.5-pro
            - OPENAI: Uses gpt-4o with fence_output=True

        The extraction result is also stored in self.result for
        subsequent operations like displayEntitiesWithPosition().
        """
        # Validate that all required configuration is set.
        if not self.prompt or not self.input_text or not self.examples:
            raise ValueError(
                f"We need prompt, input text and examples to do an extraction: "
                f"prompt: {self.prompt}, input_text: {self.input_text}, "
                f"examples: {self.examples}"
            )

        self.logger.info("extractor will commence lang extract...")

        # Execute extraction with the appropriate LLM provider.
        # Each provider has slightly different configuration options.
        if self.model == "GEMINI":
            self.result = lx.extract(
                text_or_documents=self.input_text,
                prompt_description=self.prompt,
                examples=self.examples,
                model_id="gemini-2.5-pro",
            )
        elif self.model == "OPENAI":
            self.result = lx.extract(
                text_or_documents=self.input_text,
                prompt_description=self.prompt,
                examples=self.examples,
                model_id="gpt-4o",
                api_key=os.environ.get("OPENAI_API_KEY"),
                fence_output=True,  # Helps with parsing structured output
                use_schema_constraints=False,
            )
        else:
            raise ValueError(f"The model -- {self.model} -- is invalid")

        self.logger.info("extraction completed...")
        return self.result

    def displayEntitiesWithPosition(self) -> None:
        """
        Log extracted entities with their character positions.

        Displays each extracted entity with its class, text, and
        position in the original input (if available). Useful for
        debugging and reviewing extraction results.

        Notes
        -----
        Must be called after extract(). Character positions allow
        mapping extractions back to the source text for verification.

        Example output:
            Entities with position:
            Input: John works at Acme Corp...
            Extracted entities:
            * Company: Acme Corp (pos: 14-23)
            * Job_title: developer (pos: 30-39)
        """
        self.logger.info("Entities with position: \n")
        self.logger.info(f"Input: {self.input_text}\n")
        self.logger.info("Extracted entities:")

        for entity in self.result.extractions:
            position_info = ""
            if entity.char_interval:
                start, end = (
                    entity.char_interval.start_pos,
                    entity.char_interval.end_pos,
                )
                position_info = f" (pos: {start}-{end})"
            self.logger.info(
                f"* {entity.extraction_class.capitalize()}: "
                f"{entity.extraction_text}{position_info}"
            )

    def saveResults(self, fileName: str) -> str:
        """
        Save extraction results to a JSONL file.

        Creates an annotated document file that can be used for
        further processing, model training, or archival.

        Parameters
        ----------
        fileName : str
            Base name for the output file (without extension).
            File will be saved to test_output/{fileName}.jsonl

        Returns
        -------
        str
            Path to the saved file.

        Examples
        --------
            >>> path = extractor.saveResults("interview_extractions")
            >>> print(f"Saved to: {path}")
            Saved to: test_output/interview_extractions.jsonl
        """
        lx.io.save_annotated_documents([self.result], output_name=f"{fileName}.jsonl")
        self.logger.info(f"Results saved to test_output/{fileName}.jsonl")
        return f"test_output/{fileName}.jsonl"

    def createHTMLResults(self, fileName: str) -> str:
        """
        Generate an interactive HTML visualization of extractions.

        Creates a visual representation of the extraction results
        that highlights entities in the source text. Useful for
        review and presentation.

        Parameters
        ----------
        fileName : str
            Base name for the output file (without extension).
            File will be saved to test_output/{fileName}.html

        Returns
        -------
        str
            Path to the saved HTML file.

        Examples
        --------
            >>> path = extractor.createHTMLResults("interview_viz")
            >>> print(f"Open in browser: {path}")
            Open in browser: test_output/interview_viz.html

        Notes
        -----
        The HTML file is self-contained and can be viewed in any
        web browser without requiring a server.
        """
        html_content = lx.visualize(self.result)
        with open(f"test_output/{fileName}.html", "w") as f:
            f.write(html_content)
        self.logger.info(f"html visualization saved to test_output/{fileName}.html")
        return f"test_output/{fileName}.html"
