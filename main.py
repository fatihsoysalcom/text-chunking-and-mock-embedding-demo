import textwrap
import random

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Splits a long text into smaller, overlapping chunks.
    This simulates the 'chunking' process in RAG.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks

def mock_embed_text(text_chunk: str, vector_dim: int = 768) -> list[float]:
    """
    Generates a mock embedding (vector) for a given text chunk.
    In a real RAG system, this would use a pre-trained embedding model
    to convert text into a dense numerical representation that captures
    its semantic meaning. Here, we use random numbers to demonstrate
    the *concept* of a vector output.
    """
    # For demonstration, we generate a list of random floats.
    # A real embedding model would produce meaningful, fixed-size vectors.
    return [random.uniform(-1, 1) for _ in range(vector_dim)]

if __name__ == "__main__":
    document_text = """
    Büyük dil modelleri (LLM'ler) günümüzde bilgiye erişim ve onu işleme şeklimizi kökten değiştiriyor. Ancak bu modellerin kendi eğitim verileriyle sınırlı kalması, güncel veya özel bilgilere erişimini kısıtlayabiliyor. İşte bu noktada Retrieval-Augmented Generation (RAG) mimarisi devreye giriyor. Peki, RAG'ın temelini oluşturan belgeleri anlamlı parçalara bölme (chunking) ve bu parçaları makine dilinin anlayabileceği vektörlere dönüştürme (embedding) süreci nasıl işliyor? Bu makalede, RAG uygulamalarının kalbinde yer alan bu iki kritik adımı, adım adım ve bol örnekle inceleyeceğiz.

    Günlük hayatta karşılaştığımız bilgi yığınları - e-postalar, raporlar, web sayfaları, teknik dokümantasyonlar - LLM'ler için adeta birer hazine. Ancak bu hazineyi LLM'lerin anlayabileceği, sorgulanabilir ve yanıt üretmede kullanabileceği bir forma dönüştürmek, basit bir kopyala-yapıştır işlemiyle mümkün değil. İşte bu dönüştürme sürecinin ilk ve en önemli adımları chunking ve embedding. Bu iki teknoloji, belgelerdeki bilgiyi sadece metin olarak değil, aynı zamanda anlamsal olarak da LLM'lere sunmamızı sağlıyor. Bu makalede, bu süreci sıfırdan başlayarak, teknik detaylarına inerek ve gerçek dünya senaryolarıyla pekiştirerek ele alacağız. Amacımız, RAG sistemlerinin nasıl çalıştığını anlamak ve kendi uygulamalarınızda bu teknolojileri etkin bir şekilde kullanabilmeniz için sağlam bir temel oluşturmaktır.
    """

    # Clean up the text a bit (remove excessive newlines and leading/trailing whitespace)
    document_text = textwrap.dedent(document_text).strip()

    # --- Configuration for Chunking and Embedding --- 
    CHUNK_SIZE = 200  # Number of characters per chunk
    CHUNK_OVERLAP = 50 # Number of characters to overlap between chunks
    VECTOR_DIMENSION = 5 # Reduced for readability in output; real models often use 768+ dimensions

    print(f"Original Document Length: {len(document_text)} characters\n")
    print(f"Chunking with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}\n")

    # --- Step 1: Chunking the document --- 
    # This process breaks down the large document into smaller, manageable pieces.
    # Each chunk is intended to be semantically coherent on its own.
    chunks = chunk_text(document_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Generated {len(chunks)} chunks.\n")

    # --- Step 2: Embedding each chunk --- 
    # Each chunk is then converted into a numerical vector (embedding).
    # These vectors capture the semantic meaning of the text and can be used
    # for similarity search (e.g., finding relevant chunks for a query).
    embedded_chunks = []
    for i, chunk in enumerate(chunks):
        embedding = mock_embed_text(chunk, VECTOR_DIMENSION)
        embedded_chunks.append((chunk, embedding))
        print(f"--- Chunk {i+1} ---")
        print(f"Text: \"{chunk[:70]}...\" (Length: {len(chunk)} characters)")
        print(f"Embedding (first {VECTOR_DIMENSION} dims): {embedding}\n")

    print("--- Demonstration Complete ---")
    print("In a real RAG system, these embedded chunks would be stored in a vector database,")
    print("ready to be retrieved based on a user's query embedding.")
