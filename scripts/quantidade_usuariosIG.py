import ijson

# Caminho do arquivo
caminho = "C:/TCC2/data/posts.json"

total_pares = 0
pares_unicos = set()

with open(caminho, "r", encoding="utf-8") as f:
    objetos = ijson.items(f, "item")
    for post in objetos:
        conta = post.get("account", {})
        nome = conta.get("name")
        handle = conta.get("handle")
        if nome and handle:
            total_pares += 1
            pares_unicos.add((handle.strip().lower(), nome.strip().lower()))

print(f"🔢 Total de pares (handle + name): {total_pares}")
print(f"🧮 Pares únicos (sem repetição): {len(pares_unicos)}")
