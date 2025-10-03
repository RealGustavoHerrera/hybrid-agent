from dotenv import load_dotenv
import langextract as lx
import os
import textwrap
import logging

load_dotenv()


class LangExtractor:

    def __init__(self, model):
        self.logger = logging.getLogger(__name__)
        match model:
            case "GEMINI":
                self.logger.info("creating a GEMINI instance")
                key = os.getenv("LANGEXTRACT_API_KEY")
                if not key:
                    self.logger.info("LANGEXTRACT_API_KEY not set")
                    return False
                self.model = model

            case "OPENAI":
                self.logger.info("creating an OPENAI instance")
                key = os.getenv("OPENAI_API_KEY")
                if not key:
                    self.logger.info("OPENAI_API_KEY not set")
                    return False
                self.model = model
            case _:
                raise ValueError(f"model {model} must be one of `GEMINI` or `OPENAI`")

    def setPrompt(self, prompt):
        self.prompt = textwrap.dedent(prompt)

    def setInputText(self, inputText):
        self.input_text = inputText

    def setExamples(self, examples):
        self.examples = examples

    def extract(self):
        if not self.prompt or not self.input_text or not self.examples:
            raise ValueError(
                f"We need prompt, input text and examples to do an extraction: prompt: {self.prompt}, input_text: {self.input_text}, examples: {self.examples}"
            )

        self.logger.info(f"extractor will commence lang extract...")
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
                model_id="gpt-4o",  # Automatically selects OpenAI provider
                api_key=os.environ.get("OPENAI_API_KEY"),
                fence_output=True,
                use_schema_constraints=False,
            )
        else:
            raise ValueError(f"The model -- {self.model} -- is invalid")

        self.logger.info(f"extraction completed...")
        return self.result

    def displayEntitiesWithPosition(self):
        # Display entities with positions
        self.logger.info(f"Entities with position: \n")
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
                f"• {entity.extraction_class.capitalize()}: {entity.extraction_text}{position_info}"
            )

    def saveResults(self, fileName):
        lx.io.save_annotated_documents([self.result], output_name=f"{fileName}.jsonl")
        self.logger.info(f"Results saved to test_output/{fileName}.jsonl")
        return f"test_output/{fileName}.jsonl"

    def createHTMLResults(self, fileName):
        # Generate the interactive visualization from the file
        html_content = lx.visualize(self.result)
        with open(f"test_output/{fileName}.html", "w") as f:
            f.write(html_content)
        self.logger.info(f"html visualization saved to test_output/{fileName}.html")
        return f"test_output/{fileName}.html"
