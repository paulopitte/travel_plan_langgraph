from typing import TypedDict, Annotated, List
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage

## Importando as variaveis de ambiente no caso 'OPENAI_API_KEY'
from dotenv import load_dotenv
load_dotenv()


## Criando o LLM 
llm = ChatOllama(model="llama3.2", temperature=0.7, max_tokens=500, verbose=True)
 

## Criando o Prompt de sistema (agente principal)
itinerary_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        template="Você é um assistente de viagem útil, simpatico e profissional porém com um toque de criatividade e humor. Crie um itinerário de viagem de HUM DIA para {city} com base nos \
        interesses do usuário: {interests}. Forneça um itinerário breve e com marcadores."),
    HumanMessage(content="Crie um itinerário para minha viagem de hum dia apenas(bate e volta).")]
)

 
## Funções dos agentes (serão os nós dos grafos)

## Primeiramente criando o State:
class PlannerState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    city: str
    interests: List[str]
    itinerary: str


## Função que recebe a entrada do usuário "Nó de entrada".
def input_city(state: PlannerState):
    user_message = input("Por favor, insira a cidade que deseja visitar em sua viagem de hum dia: ")
    return {
    "city": user_message,
    "messages": state["messages"] + [HumanMessage(content=user_message)]
    }


## Função que recebe os interesses adicionais "Nó intermediário":
def input_interests(state: PlannerState):
    user_message = input("Quais são seus interesses para a viagem? (ex: museus, parques, gastronomia): ")
    interests_list = [interest.strip() for interest in user_message.split(",")]
    return {
        "interests": interests_list,
        "messages": state["messages"] + [HumanMessage(content=user_message)]
    }


## Função que cria o itinerário "Nó finalizador":
def create_itinerary(state: PlannerState):
    print(f"Criando um itinerário para {state['city']} com base em interesses: {', '.join(state['interests'])}...")
    response = llm.invoke(itinerary_prompt.format_messages(city=state['city'], interests=", ".join(state['interests'])))
    print("========================================")
    print("\n### Itinerário Final:")
    print("========================================")
    print(response.content)
    print("========================================")
    return {
        "messages": state['messages'] + [("ai", response.content)],
        "itinerary": response.content,
    }

### Grafo......Ordem básica é: Estado, Nós, Arestas, compilação:

# 1) Primeiro indique o State:
builder = StateGraph(PlannerState)


# 2) Entrada do fluxo:
builder.add_edge(START, "input_city")

# 3) Nós, lembrando que ("nome do nó", "função a ser executada"):
builder.add_node("input_city", input_city)
builder.add_node("input_interests", input_interests)
builder.add_node("create_itinerary", create_itinerary)

# 4) Arestas:
builder.add_edge("input_city", "input_interests")
builder.add_edge("input_interests", "create_itinerary")
builder.add_edge("create_itinerary", END)

# 5) Compile e gere o grafo:
graph = builder.compile()

while True:
    print("Você deseja planejar um itinerário de 1 dia para sua Viagem(bate-volta)?")
    user_input = input("Resposta (Sim ou Não):")
    if user_input.lower() in ["quit", "exit", "q", "n", "não", "nao"]:
        print("Tchau Tchau!")
        break
    for event in graph.stream({"messages": [("user", user_input)]}):
        pass