import langextract as lx
from extract_graph.extractors.langextractor import LangExtractor


class ExtractorFocusOnWork(LangExtractor):
    def __init__(self, model):
        super().__init__(model)
        self.prompt = """
            Extract the work tenures.
            For each work tenure:
            - Identify the name of the company or employer.
            - Identify the position or job title held.
            - Identify the technologies used.
            - Identify the responsibilities held.
        """

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
