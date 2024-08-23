import re

import tqdm
from langchain_openai import ChatOpenAI
from modules.models.paper import Paper
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Union, List
import codecs
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pathlib import Path
import json


class SingleDatasetDetails(BaseModel):
    dataset_name: str = Field(
        description='论文中提到的数据集的名字'
    )
    dataset_usage: Union[None, str] = Field(
        description='论文中提到这个数据集的作用。如果没提到返回None'
    )
    dataset_source: Union[None, str] = Field(
        description='论文中提到该数据集的数据源。例如：Huggingface等； 如果没提到返回None'
    )
    reference_index: Union[None, str] = Field(
        description='引用中的id，如果找不到返回None'
    )


class PaperSummary(BaseModel):
    paper_target: str = Field(
        description='论文想要研究、验证、解决的目标'
    )
    paper_mentioned_datasets: List[SingleDatasetDetails] = Field(
        description='论文中提到的使用的数据集列表。'
    )


def english_stringfy_string(input):
    return re.sub('，', ', ', input)


def format_instructions_2_chinese(format_instruction_string):
    return codecs.decode(format_instruction_string, 'unicode_escape')


def extract_dataset_usage(paper_ins: Paper, paper_field=None, model_name='glm-4'):
    llm_ins = ChatOpenAI(temperature=0.95,
                         model=model_name,
                         openai_api_key="7d020833a52b08e7251707288af8d20d.JmuseA1s6dTDSyt7",
                         openai_api_base="https://open.bigmodel.cn/api/paas/v4/")
    parser = PydanticOutputParser(pydantic_object=PaperSummary)
    retry_parser = OutputFixingParser.from_llm(parser=parser, llm=llm_ins)
    format_instruction = format_instructions_2_chinese(parser.get_format_instructions())
    prompt = f"""# Role: 你是一个非常有经验的学者，你非常擅长阅读{paper_field if paper_field else ''}论文并从中提取出有用的信息。
# Task: 
我会提供你一篇论文，我需要你帮我从内容中提取出论文中提到的使用到的数据集及这个数据集相关的信息。同时我还需要你帮我总结改论文本身的一些信息。
请根据我的要求返回我JSON格式的结果，具体要求如下：
{format_instruction},

# Paper Content:
{paper_ins.all_text}

# Output
YOUR ANSWER(请返回中文结果用JSON格式):
"""
    res_raw = llm_ins.invoke(prompt)
    res_content = res_raw.content
    details = retry_parser.parse(res_content)
    return json.loads(details.json()), res_raw.response_metadata


if __name__ == "__main__":
    paper_dir = r'W:\Personal\arxiv_daily\Small_molecule_dataset\papers'
    from loguru import logger
    llm_model = 'glm-4-long'
    # paper = Path(
    #     paper_dir) / '3M-Diffusion Latent Multi-Modal Diffusion for Text-Guided Generation of Molecular Graphs.pdf'
    # paper_ins = Paper(str(paper))
    import glob
    if (Path(paper_dir).parent / f'dataset_extraction_{llm_model}.json').exists():
        with open(Path(paper_dir).parent / f'dataset_extraction_{llm_model}.json', 'r', encoding='utf-8') as f:
            current_results = json.load(f)
            finished = current_results.keys()
    else:
        finished = []
        current_results = {}
    papers = glob.glob(str(Path(paper_dir) / '*.pdf'))
    for paper in tqdm.tqdm(papers):
        logger.info(f"Working on {paper}")
        if str(Path(paper).name) in finished:
            logger.success("Finished, skipped.")
            continue
        try:
            paper_ins = Paper(paper)
            res, response_metadata = extract_dataset_usage(paper_ins, 'Small molecule', model_name=llm_model)
        except:
            res = {'status': 'Failed'}
            response_metadata = {}
        current_results[str(Path(paper).name)] = res
        logger.success(paper)
        logger.success(res)
        with open(Path(paper_dir).parent / f'dataset_extraction_{llm_model}.json', 'w', encoding='utf-8') as f:
            json.dump(current_results, f, indent=4, ensure_ascii=False)
