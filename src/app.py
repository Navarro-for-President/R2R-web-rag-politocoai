"""A simple example to demonstrate the usage of `WebRAGPipeline`."""
import logging
from typing import Optional

from r2r.core import (
    LLMProvider,
    LoggingDatabaseConnection,
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


# Creates a pipeline using the `WebRAGPipeline` implementation
app = E2EPipelineFactory.create_pipeline(
    rag_pipeline_impl=WebRAGPipeline,
    config_path='config.json'
)
