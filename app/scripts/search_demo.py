from app.services.searcher import ImageSearcher

searcher = ImageSearcher()

query = input("Enter a search query: ")
results = searcher.search(query, top_k=5)

print(f"\nTop {len(results)} results for: '{query}'\n")
for i, r in enumerate(results, 1):
    print(f"{i}. score={r['score']:.4f}")
    print(f"   image: {r['image_path']}")
    print(f"   caption: {r['caption']}\n")
