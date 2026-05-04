import logging
from langchain_core.documents import Document
import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_anthropic import ChatAnthropic
from langchain_fireworks import ChatFireworks
from langchain_aws import ChatBedrock
from langchain_community.chat_models import ChatOllama
import boto3
import google.auth
from src.shared.llm_graph_builder_exception import LLMGraphBuilderException
from typing import List
from langchain_core.callbacks.manager import CallbackManager
from src.shared.common_fn import UniversalTokenUsageHandler, get_value_from_env

def get_llm(model: str):
    """Retrieve the specified language model based on the model name."""
    model = model.upper().replace('.', '_').strip()
    env_key = f"LLM_MODEL_CONFIG_{model}"
    env_value = get_value_from_env(env_key)

    if not env_value:
        err = f"Environment variable '{env_key}' is not defined as per format or missing"
        logging.error(err)
        raise Exception(err)
    
    logging.info("Model: {}".format(env_key))
    callback_handler = UniversalTokenUsageHandler()
    callback_manager = CallbackManager([callback_handler])
    try:
        if "GEMINI" in model:
            model_name = env_value
            credentials, project_id = google.auth.default()
            llm = ChatVertexAI(
                model_name=model_name,
                credentials=credentials,
                project=project_id,
                temperature=0,
                callbacks=callback_manager,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                },
            
            )
        elif "OPENAI" in model:
            model_name, api_key = env_value.split(",")
            if "MINI" in model:
                llm= ChatOpenAI(
                api_key=api_key,
                model=model_name,
                callbacks=callback_manager,
                )
            else:
                llm = ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=0,
                callbacks=callback_manager,
                )

        elif "AZURE" in model:
            model_name, api_endpoint, api_key, api_version = env_value.split(",")
            llm = AzureChatOpenAI(
                api_key=api_key,
                azure_endpoint=api_endpoint,
                azure_deployment=model_name,  # takes precedence over model parameter
                api_version=api_version,
                temperature=0,
                max_tokens=None,
                timeout=None,
                callbacks=callback_manager,
            )

        elif "ANTHROPIC" in model:
            model_name, api_key = env_value.split(",")
            llm = ChatAnthropic(
                api_key=api_key, model=model_name, temperature=0, timeout=None,callbacks=callback_manager, 
            )

        elif "FIREWORKS" in model:
            model_name, api_key = env_value.split(",")
            llm = ChatFireworks(api_key=api_key, model=model_name,callbacks=callback_manager)

        elif "GROQ" in model:
            model_name, base_url, api_key = env_value.split(",")
            llm = ChatGroq(api_key=api_key, model_name=model_name, temperature=0,callbacks=callback_manager)

        elif "BEDROCK" in model:
            model_name, aws_access_key, aws_secret_key, region_name = env_value.split(",")
            bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=region_name,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
            )

            llm = ChatBedrock(
                client=bedrock_client,region_name=region_name, model_id=model_name, model_kwargs=dict(temperature=0),callbacks=callback_manager, 
            )

        elif "OLLAMA" in model:
            model_name, base_url = env_value.split(",")
            llm = ChatOllama(base_url=base_url, model=model_name,callbacks=callback_manager)

        elif "DIFFBOT" in model:
            raise LLMGraphBuilderException(
                "Diffbot is no longer supported. Pick a different model — see release notes."
            )

        else:
            model_name, api_endpoint, api_key = env_value.split(",")
            llm = ChatOpenAI(
                api_key=api_key,
                base_url=api_endpoint,
                model=model_name,
                temperature=0,
                callbacks=callback_manager,
            )
    except Exception as e:
        err = f"Error while creating LLM '{model}': {str(e)}"
        logging.error(err)
        raise Exception(err)
 
    logging.info(f"Model created - Model Version: {model}")
    return llm, model_name, callback_handler

def get_llm_model_name(llm):
    """Extract name of llm model from llm object"""
    for attr in ["model_name", "model", "model_id"]:
        model_name = getattr(llm, attr, None)
        if model_name:
            return model_name.lower()
    logging.info("Could not determine model name; defaulting to empty string")
    return ""

def get_combined_chunks(chunkId_chunkDoc_list, chunks_to_combine):
    combined_chunk_document_list = []
    combined_chunks_page_content = [
        "".join(
            document["chunk_doc"].page_content
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        )
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]
    combined_chunks_ids = [
        [
            document["chunk_id"]
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        ]
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]

    for i in range(len(combined_chunks_page_content)):
        combined_chunk_document_list.append(
            Document(
                page_content=combined_chunks_page_content[i],
                metadata={"combined_chunk_ids": combined_chunks_ids[i]},
            )
        )
    return combined_chunk_document_list

def get_chunk_id_as_doc_metadata(chunkId_chunkDoc_list):
    combined_chunk_document_list = [
       Document(
           page_content=document["chunk_doc"].page_content,
           metadata={"chunk_id": [document["chunk_id"]]},
       )
       for document in chunkId_chunkDoc_list
   ]
    return combined_chunk_document_list
      

async def get_graph_from_llm(
    model,
    chunkId_chunkDoc_list,
    allowedNodes,
    allowedRelationship,
    chunks_to_combine,
    additional_instructions=None,
    schemaSpec=None,
):
    """Run entity/relation extraction via neo4j-graphrag and return LangChain
    GraphDocuments + token usage.

    Inputs:
      - schemaSpec (preferred): JSON-encoded typed SchemaSpec from the
        frontend's buildSchemaSpec helper.
      - allowedNodes / allowedRelationship: legacy comma-separated fallback
        used when schemaSpec is unset (preserved so already-deployed
        frontends still work during a frontend deploy lag).
    """
    try:
        from src.graphrag.extractor import derive_schema_spec, extract_via_graphrag

        combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list, chunks_to_combine)
        logging.info(f"Combined {len(combined_chunk_document_list)} chunks")

        schema_spec_obj = _parse_schema_spec(schemaSpec)

        if allowedNodes:
            allowed_nodes = [node.strip() for node in allowedNodes.split(',') if node.strip()]
        else:
            allowed_nodes = []

        allowed_relationships = []
        if allowedRelationship:
            items = [item.strip() for item in allowedRelationship.split(',') if item.strip()]
            if len(items) % 3 != 0:
                raise LLMGraphBuilderException("allowedRelationship must be a multiple of 3 (source, relationship, target)")
            for i in range(0, len(items), 3):
                source, relation, target = items[i:i + 3]
                if allowed_nodes and (source not in allowed_nodes or target not in allowed_nodes):
                    raise LLMGraphBuilderException(
                        f"Invalid relationship ({source}, {relation}, {target}): "
                        f"source or target not in allowedNodes"
                    )
                allowed_relationships.append((source, relation, target))

        spec = derive_schema_spec(
            schema_spec_obj,
            allowed_nodes,
            allowed_relationships,
            None,
            None,
        )

        graph_document_list, token_usage = await extract_via_graphrag(
            model=model,
            combined_chunk_document_list=combined_chunk_document_list,
            schema_spec=spec,
            additional_instructions=additional_instructions,
        )
        logging.info(f"Generated {len(graph_document_list)} graph documents")
        return graph_document_list, token_usage
    except Exception as e:
        logging.error(f"Error in get_graph_from_llm: {e}", exc_info=True)
        raise LLMGraphBuilderException(f"Error in getting graph from llm: {e}")


def _parse_schema_spec(raw):
    """Parse a JSON-encoded SchemaSpec from /extract; return SchemaSpec | None."""
    if not raw:
        return None
    try:
        from src.graphrag.schema_model import SchemaSpec
        return SchemaSpec.model_validate_json(raw)
    except Exception as e:
        logging.warning(f"Could not parse schemaSpec: {e}")
        return None
