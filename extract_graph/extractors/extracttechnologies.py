"""
Work History Entity Extractor
=============================

This module extracts work history information from unstructured text such as
interview transcripts or resumes. It extends LangExtractor with a pre-configured
prompt and few-shot examples for work history extraction.

What It Does
------------
- Identifies companies and employers mentioned in text
- Extracts job titles and positions held
- Finds technologies, tools, and frameworks used
- Captures responsibilities and duties described

Entity Classes Extracted
------------------------
- **company**: Company or employer names
- **job_title**: Position or job title held
- **technology**: Technologies, tools, frameworks used
- **responsibilities**: Job responsibilities and duties

Example Usage
-------------
    from extract_graph.extractors.extracttechnologies import ExtractorFocusOnWork

    # Create extractor
    extractor = ExtractorFocusOnWork("OPENAI")

    # Set input text (interview transcript, resume, etc.)
    extractor.setInputText(interview_transcript)

    # Extract entities
    result = extractor.extract()

    # View results
    for entity in result.extractions:
        print(f"{entity.extraction_class}: {entity.extraction_text}")

    # Save results
    extractor.saveResults("interview_extractions")

Graph Construction
------------------
Extracted entities can be used to build a knowledge graph:

    Person --[WORKED_AT]--> Company
    Person --[HAS_TITLE]--> JobTitle
    Person --[USED_TECH]--> Technology
    Job --[REQUIRES]--> Technology

See Also
--------
- extract_graph.extractors.langextractor : Base class for extractors
- Apache AGE : Graph database for storing extracted relationships
"""

import langextract as lx
from extract_graph.extractors.langextractor import LangExtractor


class ExtractorFocusOnWork(LangExtractor):
    """
    Specialized extractor for work history and technology information.

    This extractor is pre-configured with prompts and examples optimized
    for extracting professional information from interview transcripts,
    resumes, and similar documents.

    The extractor identifies four types of entities:
    - Companies/employers
    - Job titles/positions
    - Technologies/tools used
    - Responsibilities held

    Attributes
    ----------
    prompt : str
        Pre-configured extraction instructions for work history.
    examples : List[lx.data.ExampleData]
        Few-shot examples demonstrating expected extractions.

    Parameters
    ----------
    model : str
        LLM provider to use. Must be 'GEMINI' or 'OPENAI'.

    Examples
    --------
    Extract from interview transcript:

        >>> extractor = ExtractorFocusOnWork("OPENAI")
        >>> extractor.setInputText(transcript)
        >>> result = extractor.extract()
        >>> companies = [e for e in result.extractions
        ...              if e.extraction_class == "company"]

    Notes
    -----
    The few-shot examples are critical for extraction quality.
    Consider customizing examples for your specific domain if
    the default examples don't match your data well.

    See Also
    --------
    LangExtractor : Base class providing extraction workflow
    """

    def __init__(self, model: str):
        """
        Initialize the work history extractor.

        Sets up the extraction prompt and few-shot examples for
        work history entity extraction.

        Parameters
        ----------
        model : str
            LLM provider to use. Must be 'GEMINI' or 'OPENAI'.

        Raises
        ------
        ValueError
            If model is invalid or required API key is not set.
        """
        super().__init__(model)

        # Domain-specific prompt for work history extraction.
        # The prompt guides the LLM on what entities to extract
        # and how to categorize them.
        self.prompt = """
            Extract the work tenures.
            For each work tenure:
            - Identify the name of the company or employer.
            - Identify the position or job title held.
            - Identify the technologies used.
            - Identify the responsibilities held.
        """

        # Few-shot examples demonstrating expected extractions.
        # These examples significantly improve extraction accuracy
        # by showing the LLM the expected output format.
        self.examples = [
            lx.data.ExampleData(
                text=(
                    """
                            PRIVATE CONTENT (Interview to FDLR)

                        """
                ),
                extractions=[
                    lx.data.Extraction(
                        extraction_class="company", extraction_text="Techint"
                    ),
                    lx.data.Extraction(
                        extraction_class="company", extraction_text="Paychex corp USA"
                    ),
                    lx.data.Extraction(
                        extraction_class="company", extraction_text="Wenance"
                    ),
                    lx.data.Extraction(
                        extraction_class="company",
                        extraction_text="Government of Buenos Aires City",
                    ),
                    lx.data.Extraction(
                        extraction_class="company", extraction_text="Coremation"
                    ),
                    lx.data.Extraction(
                        extraction_class="company",
                        extraction_text="Software del Plata.",
                    ),
                    lx.data.Extraction(
                        extraction_class="job_title",
                        extraction_text="Software Engineer",
                    ),
                    lx.data.Extraction(
                        extraction_class="job_title", extraction_text="Main Architect"
                    ),
                    lx.data.Extraction(
                        extraction_class="job_title", extraction_text="Software Manager"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="Java"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="Mongo"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="Springboot"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="RESTful API"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="Kafka"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="Kubernetes"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="Jenkins"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="Jenkins"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="Lombok"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="SonarQube"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="AWS"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="EKS"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="Lambdas"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="DynamoDB"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="S3"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="MongoDB"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="PostgreSQL"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="GraphQL"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="Redis"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="OpenSearch"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="SNS"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="SQS"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="Node.js"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="Angular"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="MySQL"
                    ),
                    lx.data.Extraction(
                        extraction_class="technology", extraction_text="Hibernate"
                    ),
                    lx.data.Extraction(
                        extraction_class="responsibilities",
                        extraction_text="part of a team of senior developers",
                    ),
                    lx.data.Extraction(
                        extraction_class="responsibilities", extraction_text="coaching"
                    ),
                    lx.data.Extraction(
                        extraction_class="responsibilities", extraction_text="mentoring"
                    ),
                    lx.data.Extraction(
                        extraction_class="responsibilities",
                        extraction_text="architectural design",
                    ),
                    lx.data.Extraction(
                        extraction_class="responsibilities",
                        extraction_text="software design",
                    ),
                    lx.data.Extraction(
                        extraction_class="responsibilities",
                        extraction_text="integration",
                    ),
                    lx.data.Extraction(
                        extraction_class="responsibilities",
                        extraction_text="automation",
                    ),
                    lx.data.Extraction(
                        extraction_class="responsibilities", extraction_text="support"
                    ),
                    lx.data.Extraction(
                        extraction_class="responsibilities", extraction_text="training"
                    ),
                ],
            ),
        ]
