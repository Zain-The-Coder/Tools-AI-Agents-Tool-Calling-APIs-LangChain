from langchain_community.tools import tool

@tool
def greet_user (name : str) -> str :
    """Generate a greeting message or a user"""

    return f"Hello {name} , Welcome to this AI world"


result = greet_user.invoke({"name" : "zain"})
print(result)