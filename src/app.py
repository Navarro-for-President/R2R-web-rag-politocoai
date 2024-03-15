"""A simple example to demonstrate the usage of `WebRAGPipeline`."""
import json
import logging
from typing import Generator, Optional

from r2r.core import (
    GenerationConfig,
    LLMProvider,
    LoggingDatabaseConnection,
    RAGPipeline,
    VectorDBProvider,
    log_execution_to_db,
)
from r2r.embeddings import OpenAIEmbeddingProvider
from r2r.integrations import SerperClient
from r2r.main import E2EPipelineFactory
from r2r.pipelines import BasicRAGPipeline

logger = logging.getLogger(__name__)

class WebRAGPipeline(BasicRAGPipeline):
    def __init__(
        self,
        llm: LLMProvider,
        db: VectorDBProvider,
        embedding_model: str,
        embeddings_provider: OpenAIEmbeddingProvider,
        logging_connection: Optional[LoggingDatabaseConnection] = None,
        system_prompt: Optional[str] = None,
        task_prompt: Optional[str] = None,
    ) -> None:
        logger.debug(f"Initalizing `WebRAGPipeline`.")
        super().__init__(
            llm=llm,
            logging_connection=logging_connection,
            db=db,
            embedding_model=embedding_model,
            embeddings_provider=embeddings_provider,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
        )
        self.serper_client = SerperClient()

    def transform_query(self, query: str) -> str:
        # Placeholder for query transformation - A unit transformation.
        return query

    @log_execution_to_db
    def search(
        self,
        transformed_query: str,
        filters: dict,
        limit: int,
        *args,
        **kwargs,
    ) -> list:
        return self.serper_client.get_raw(transformed_query, limit)

    @log_execution_to_db
    def construct_context(self, results: list) -> str:
        return self.serper_client.construct_context(results)


    def _stream_run(
        self,
        search_results: list,
        context: str,
        prompt: str,
        generation_config: GenerationConfig,
    ) -> Generator[str, None, None]:
        yield f"<{RAGPipeline.SEARCH_STREAM_MARKER}>"
        yield json.dumps(search_results)
        yield f"</{RAGPipeline.SEARCH_STREAM_MARKER}>"

        yield f"<{RAGPipeline.CONTEXT_STREAM_MARKER}>"
        yield context
        yield f"</{RAGPipeline.CONTEXT_STREAM_MARKER}>"
        yield f"<{RAGPipeline.COMPLETION_STREAM_MARKER}>"
        for chunk in self.generate_completion(prompt, generation_config):
            yield chunk
        yield f"</{RAGPipeline.COMPLETION_STREAM_MARKER}>"

# Creates a pipeline using the `WebRAGPipeline` implementation
app = E2EPipelineFactory.create_pipeline(
    rag_pipeline_impl=WebRAGPipeline,
    config=R2RConfig.load_config("config.json")
)
