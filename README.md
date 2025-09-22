# Agents, MCPs & LangGraph - A Tutorial

This repository serves as the official companion for the agentic systems tutorial presented at [GCPR 2025][gcpr]. The content focuses on two key frameworks for building robust, multi-step agents: [LangGraph][lg] and [Model Context Protocol(MCP)][mcp]. 

All code and materials are provided to facilitate a hands-on learning experience.



## Quick start
All the sessions can be run in Google Colab. Click on the icons below to open the respective notebooks in Colab and get started.

|Session|Notebook|Colab|
|:---|:---|:---:|
|Intro to langgraph| [`01_basic_langgraph_chatbot.ipynb`](./01_basic_langgraph_chatbot.ipynb) | [![Open in Colab][colab-badge]](https://colab.research.google.com/github/TuebingenAICenter/agent-tutorial/blob/main/01_basic_langgraph_chatbot.ipynb)|
|Langgraph fundamentals| [`02_chat_with_pdfs_and_youtube_scaffolded.ipynb`](./02_chat_with_pdfs_and_youtube_scaffolded.ipynb) | [![Open in Colab][colab-badge]](https://colab.research.google.com/github/TuebingenAICenter/agent-tutorial/blob/main/02_chat_with_pdfs_and_youtube_scaffolded.ipynb)|
|MCP| [`mcp_colab.ipynb`](./mcp/mcp_colab.ipynb) | [![Open in Colab][colab-badge]](https://colab.research.google.com/github/TuebingenAICenter/agent-tutorial/blob/main/mcp/mcp_colab.ipynb)|
|Orchestration| [`03_deepresearch.ipynb`](./03_deepresearch.ipynb) | [![Open in Colab][colab-badge]](https://colab.research.google.com/github/TuebingenAICenter/agent-tutorial/blob/main/03_deepresearch.ipynb)|


> [!NOTE]
> Make sure to run the first cell in each notebook to install the required packages, since every Colab notebook runs in a fresh environment.
>
> Remember to add your API keys in the `.env` file created in the first cell.


## Local setup

For those feeling more adventurous, follow the instructions below to set up the repo locally on your system

1. Clone the repo
    ```bash
        git clone https://github.com/TuebingenAICenter/agent-tutorial-dev.git
        cd agent-tutorial
    ```
2. Create and activate a virtual environment using a python manager of your choice. Install the required packages. (Instructions for `uv` and `venv` are listed below)
3. Create a copy of the `env.example` file and rename it to `.env`. Add your API keys to this file.
    


### venv
```bash
    # We do recommend using python 3.12, but any version >=3.10 should work
    python3.12 -m venv .venv
    source .venv/bin/activate
```
```bash
    pip install -r requirements.txt
```
### uv [Recommended]
```bash
    uv venv -p 3.12
    source .venv/bin/activate
    uv pip install -r requirements.txt
```





## Authors and collaborators

<table width="100%">
  <tr>
    <td align="center">
        <div style="width: 150px;">
        <img src="https://tuebingen.ai/fileadmin/user_upload/Home/News/2025/2025-02-17_Peter-Gehler/PeterGehler_landscape-856px_V3A0360.jpg" alt="Peter Gehler" width="150px;">
        <h4>
            <a href="https://tuebingen.ai">
                Peter Gehler
            </a>
        </h4>
        <sub><i>Prof. Tech Transfer @Tübingen AI Center</i></sub>
        </div>
    </td>
    <td align="center">
        <div style="width: 150px;">
        <img src="https://nc.mlcloud.uni-tuebingen.de/index.php/apps/files_sharing/publicpreview/bybeXfZaGyxJ6nX?file=/&fileId=17890935&x=1920&y=1080&a=true&etag=3af9170c15d4a2e1583ae41d9c3bc366" alt="Matthias Kümmerer" width="150px;">
        <h4>
            <a href="https://bethgelab.org/">
                Matthias Kümmerer
            </a>
        </h4>
        <sub><i>Postdoc @bethgelab</i></sub>
        </div>
    </td>
    <td align="center">
        <div style="width: 150px;">
        <img src="https://avatars.githubusercontent.com/u/97098427?v=4" alt="Linus A. Schneider" width="150px;">
        <h4>
            <a href="https://github.com/shoshinL">
                Linus A. Schneider
            </a>
        </h4>
        <sub><i>Researcher</i></sub>
        </div>
    </td>
  </tr>
  <tr>
    <td align="center">
        <div style="width: 150px;">
        <img src="https://avatars.githubusercontent.com/u/75247817?v=4" alt="Jaisidh Singh" width="150px;">
        <h4>
            <a href="https://github.com/jaisidhsingh">
                Jaisidh Singh
            </a>
        </h4>
        <sub><i>Researcher</i></sub>
        </div>
    </td>
    <td align="center">
        <div style="width: 150px;">
        <img src="https://tuebingen.ai/fileadmin/_processed_/7/a/csm_Robin-Ruff_688b07f0e2.jpg" alt="Robin" width="150px;">
        <h4>
            <a href="https://github.com/robinruff">
            Robin Ruff
            </a>
        </h4>
        <sub><i>SE @Tübingen AI Center</i></sub>
        </div>
    </td>
    <td align="center">
        <div style="width: 150px;">
        <img src="https://tuebingen.ai/fileadmin/_processed_/0/a/csm_Mamen-Thomas-Chembakasseril_596595dc7c.jpg" alt="Mamen" width="150px;">
        <h4>
            <a href="https://github.com/mtc-20">
                Mamen
            </a>
        </h4>
        <sub><i>SE @Tübingen AI Center</i></sub>
        </div>
    </td>
  </tr>
</table>




<!-- TO BE REMOVED -->
<!-- ## Dev resources
- [Google slides: Agents and how to get started with Langgraph ][1]
- [Google Colab: simple chat notebook][2]
- [PDF: OpenAI A practical guide to building agents][3]
- [Repo: Langgraph Deep Research from scratch][4]
    - [Course: ][4A]
- [Repo: Perplexica][5]
- [Repo: HuggingFace Open Deep Research][6]

## TODO
- [ ] License
- [ ] DISCLAIMER
- [ ] API Keys and environment variables
- [ ] Setup instructions and requirements



[1]: https://docs.google.com/presentation/d/1hgG-bZOhD7q1VPcLFIWIKymWp46MFxYbdtn9coP3Q2o/mobilepresent#slide=id.g35ea3d3cc5f_0_17
[2]: https://colab.research.google.com/drive/1U97KmxMIT1EBBLQqSgE3k2KcDQ484JHi?usp=sharing
[3]: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
[4]: https://github.com/langchain-ai/deep_research_from_scratch
[5]: https://github.com/ItzCrazyKns/Perplexica
[4A]: https://academy.langchain.com/courses/take/deep-research-with-langgraph/texts/67644896-getting-set-up
[6]: https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research -->

<!-- END OF TO BE REMOVED -->

<!-- LINKS -->
[gcpr]: https://www.dagm-gcpr.de/year/2025/program/
[slides]: ./slides.pdf
[lg]: https://www.langchain.com/langgraph
[mcp]: https://modelcontextprotocol.io/docs/getting-started/intro
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg