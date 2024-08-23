from modules.models.paper import Paper
from pathlib import Path
paper_dir = r'W:\Personal\arxiv_daily\Small_molecule_dataset\papers'
paper = Path(paper_dir)/'3M-Diffusion Latent Multi-Modal Diffusion for Text-Guided Generation of Molecular Graphs.pdf'
paper_ins = Paper(str(paper))
# print("HERE")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key="7d020833a52b08e7251707288af8d20d.JmuseA1s6dTDSyt7",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

print("HERE")