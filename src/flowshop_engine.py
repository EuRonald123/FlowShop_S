import numpy as np
import os

#Engine unificada para Flow Shop (Tardiness ou Makespan)
class FlowShopEngine:
    
    def __init__(self, n_jobs, n_machines, proc_times, due_dates):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.proc_times = proc_times
        self.due_dates = due_dates
        self.cfo_count = 0

    @staticmethod
    def carregar_instancia_txt(caminho_arquivo):
        if not os.path.exists(caminho_arquivo):
            raise FileNotFoundError(
                f"Instância não encontrada: {caminho_arquivo}. "
                "Verifique o path antes de rodar o experimento — "
                "não há fallback automático para dados simulados."
            )

        with open(caminho_arquivo, "r") as f:
            linhas = [l.strip() for l in f if l.strip()]

        try:
            num_jobs, num_machines = map(int, linhas[0].split())
            p_matrix = np.zeros((num_jobs, num_machines), dtype=int)

            idx = 2
            for j in range(num_jobs):
                partes = linhas[idx].split()
                for i in range(0, len(partes), 2):
                    maq, tempo = int(partes[i]), int(partes[i + 1])
                    p_matrix[j, maq] = tempo
                idx += 1

            while idx < len(linhas) and linhas[idx].lower() != "duedate":
                idx += 1
            idx += 1

            due_dates = [int(linhas[idx + k]) for k in range(num_jobs)]
        except Exception as e:
            raise ValueError(f"Erro ao processar a instância: {caminho_arquivo}. Detalhes: {e}")from e
        

        if p_matrix.shape != (num_jobs, num_machines):
            raise ValueError(f"Instância malformada: {caminho_arquivo}")

        return num_jobs, num_machines, p_matrix, np.array(due_dates)

    @staticmethod
    def calcular_lower_bound_potts(num_jobs, num_machines, p_matrix, due_dates):
        max_lateness_global = -float('inf')

        for m in range(num_machines):
            if m > 0:
                r_m = np.sum(p_matrix[:, :m], axis=1)
            else:
                r_m = np.zeros(num_jobs)

            if m < num_machines - 1:
                tempo_posterior = np.sum(p_matrix[:, m+1:], axis=1)
                d_m = due_dates - tempo_posterior
            else:
                d_m = due_dates.copy()

            p_m = p_matrix[:, m].astype(float)
            rem_p = p_m.copy()
            t = 0.0
            lateness_maquina = -float('inf')

            while np.sum(rem_p > 0) > 0:
                jobs_disponiveis = (r_m <= t) & (rem_p > 0)

                if not np.any(jobs_disponiveis):
                    proximas_liberacoes = r_m[(r_m > t) & (rem_p > 0)]
                    t = np.min(proximas_liberacoes)
                    continue

                indices_validos = np.where(jobs_disponiveis)[0]
                job_escolhido = indices_validos[np.argmin(d_m[indices_validos])]

                proximas_liberacoes = r_m[(r_m > t) & (rem_p > 0)]
                if len(proximas_liberacoes) > 0:
                    prox_release_evento = np.min(proximas_liberacoes)
                else:
                    prox_release_evento = float('inf')

                tempo_ate_terminar = rem_p[job_escolhido]
                tempo_ate_interrupcao = prox_release_evento - t
                delta_t = min(tempo_ate_terminar, tempo_ate_interrupcao)

                t += delta_t
                rem_p[job_chosen := job_escolhido] -= delta_t

                if rem_p[job_escolhido] <= 1e-7:
                    lateness_j = t - d_m[job_escolhido]
                    if lateness_j > lateness_maquina:
                        lateness_maquina = lateness_j

            if lateness_maquina > max_lateness_global:
                max_lateness_global = lateness_maquina

        return float(max_lateness_global)
    
    #Nucleo único de simulação usado por evaluate e evaluate_with_permutation
    def _simular(self, sequences):
        n, m = self.n_jobs, self.n_machines
        finish_times = np.zeros((m, n))

        current_time = 0
        for job_idx in sequences[0]:
            current_time += self.proc_times[job_idx, 0]
            finish_times[0, job_idx] = current_time

        for i in range(1, m):
            time_on_m = 0
            for job_idx in sequences[i]:
                start = max(time_on_m, finish_times[i - 1, job_idx])
                finish_times[i, job_idx] = start + self.proc_times[job_idx, i]
                time_on_m = finish_times[i, job_idx]

        return finish_times

    def _custo(self, finish_times, objective):
        if objective == "Makespan":
            return np.max(finish_times[-1, :])
        elif objective == "Tardiness":
            return np.sum(np.maximum(0, finish_times[-1, :] - self.due_dates))
        raise ValueError(f"Objective inválido: {objective}")

    def evaluate(self, keys, mode="PFS", objective="Tardiness"):
        self.cfo_count += 1
        n = self.n_jobs
        if mode == "PFS":
            sequences = [np.argsort(keys[:n])] * self.n_machines
        else:
            sequences = [np.argsort(keys[i * n:(i + 1) * n]) for i in range(self.n_machines)]
        return self._custo(self._simular(sequences), objective)

    def evaluate_with_permutation(self, permutation, objective="Tardiness"):
        self.cfo_count += 1
        sequences = [permutation] * self.n_machines
        return self._custo(self._simular(sequences), objective)