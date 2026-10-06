import os
import json
import urllib.request

def baixar_taillard():
    base_dir = "instancias_taillard"
    os.makedirs(base_dir, exist_ok=True)
    
    url_base = "https://raw.githubusercontent.com/chneau/go-taillard/master/pfsp/instances/"
    
    # As 12 dimensões de Taillard
    sizes = [
        (20, 5), (20, 10), (20, 20),
        (50, 5), (50, 10), (50, 20),
        (100, 5), (100, 10), (100, 20),
        (200, 10), (200, 20),
        (500, 20)
    ]
    
    bks_data = {}
    
    count = 1
    for n, m in sizes:
        for i in range(10):
            nome_original = f"tai{n}_{m}_{i}.fsp"
            nome_padrao = f"ta{count:03d}.txt"  # Renomeia para o padrão conhecido (ta001.txt)
            url = url_base + nome_original
            caminho_local = os.path.join(base_dir, nome_padrao)
            
            try:
                urllib.request.urlretrieve(url, caminho_local)
                
                # Lê o arquivo recém-baixado para extrair a BKS da linha 2
                with open(caminho_local, "r") as f:
                    linhas = f.readlines()
                    # A linha 2 (index 1) contém: n_jobs, n_machines, seed, upper_bound, lower_bound
                    # ex: "          20           5   873654221        1278        1232"
                    partes = linhas[1].strip().split()
                    if len(partes) >= 4:
                        bks = int(partes[3])
                        bks_data[f"ta{count:03d}"] = bks
                        
                print(f"Baixado: {nome_padrao} (BKS: {bks})")
                count += 1
            except Exception as e:
                print(f"Erro ao baixar {nome_original}: {e}")

    with open(os.path.join(base_dir, "taillard_bks.json"), "w") as f:
        json.dump(bks_data, f, indent=4)
    print("\nDownload concluído e taillard_bks.json atualizado com sucesso!")

if __name__ == "__main__":
    baixar_taillard()
