# ovbench — benchmark de LLMs em OpenVINO

Mede o desempenho de **qualquer LLM** em formato OpenVINO IR em **qualquer máquina Intel**: CPU híbrida
(P-cores + E-cores), Xeon, Core clássico, e opcionalmente iGPU e NPU. As configurações a comparar são geradas
a partir da topologia real da máquina, em vez de números fixos para um único processador.

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![backend](https://img.shields.io/badge/backend-OpenVINO-0071C5)

## Por que existe

Benchmarks de LLM costumam ser scripts presos a um modelo e a uma máquina. O `ovbench` separa as três partes:

| Parte | Como o `ovbench` trata |
|---|---|
| **Modelo** | Qualquer causal LM já convertido para OpenVINO IR (`ovbench convert` converte e quantiza modelos do Hugging Face). |
| **Hardware** | Detecta modelo da CPU, núcleos P/E, instruções (AVX2, AVX-512, VNNI, AMX), RAM e dispositivos OpenVINO. Gera os presets a partir disso. |
| **Carga de teste** | Prompts embutidos, arquivos seus (`.jsonl`, `.json`, `.txt`) e uma escada de tamanhos de entrada. |

## O que é medido

- **TTFT**: tempo até o primeiro token (inclui o prefill).
- **Decode tok/s**: tokens por segundo só na fase de decodificação.
- **Latência** média, p50 e p95, por configuração e por prompt.
- **Pool**: throughput, espera em fila e latência fim a fim sob concorrência.
- **Pico de RAM** e **tempo de carga** do modelo.
- **Formato válido** (opcional), via validador plugável.
- **Ficha do ambiente** em todo relatório: CPU, núcleos, instruções, RAM, SO e versões dos pacotes.

O `ovbench` mede **velocidade, não qualidade**. Use `--save-responses` para guardar as respostas completas e
avaliá-las manualmente.

## Instalação

Requer Python 3.10 ou superior.

```bash
git clone https://github.com/patrickufpr/llmbench-openvino.git
cd llmbench-openvino
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate

pip install -e ".[openvino]"   # núcleo + OpenVINO, optimum-intel, transformers, torch
pip install -e ".[dev]"        # opcional: pytest, para rodar os testes
```

Sem o extra `openvino` só funciona o backend `fake`, usado para testes.

## Início rápido

```bash
# 1. Veja o que foi detectado e os presets gerados para esta máquina
ovbench info

# 2a. Baixe um modelo que já está em OpenVINO IR (ex.: os da organização OpenVINO no Hugging Face)
ovbench download --repo-id OpenVINO/Qwen2.5-1.5B-Instruct-int8-ov

# 2b. ...ou converta e quantize um modelo qualquer do Hugging Face
ovbench convert --model-id Qwen/Qwen2.5-0.5B-Instruct --out modelos/qwen25-0.5b --weight-format int8

# 3. Rode o benchmark
ovbench run --model modelos/qwen25-0.5b --repetitions 3 --fixed-length 128
```

Os resultados são gravados em `resultados/` como `.json` (tudo, por amostra), `.md` (tabela legível) e
`.csv` (uma linha por configuração).

### Exemplo de saída

Medido numa Core i7-1165G7 (4 núcleos, 8 threads), com um Qwen2.5-0.5B em INT8, 8 repetições, 128 tokens
gerados, com `--interleave --cooldown 5`:

| Config | Lat. média (ms) | TTFT (ms) | Decode tok/s | Pico RAM (MB)* |
|---|---|---|---|---|
| default | 4221 | 137 | 31,6 | 4959 |
| latency_fisicos | 3892 | 133 | 34,6 | 4959 |
| latency_logicos | 3883 | 132 | 34,8 | 4959 |
| kv_cache_u8 | 3956 | 135 | 34,0 | 4959 |

\* No modo `--interleave` o pico de RAM é o do conjunto (todos os presets carregados ao mesmo tempo).

Nessa máquina os presets com threads definidas ficam cerca de 9% acima do `default`, e as diferenças entre eles
estão dentro do ruído. Os números dependem de máquina, temperatura e modelo; veja a seção sobre medições
estáveis antes de tirar conclusões de diferenças pequenas.

## Topologia da CPU

O `ovbench` lê a topologia (núcleos P/E, threads) do sistema. **Em WSL2 e máquinas virtuais o sistema esconde
essa informação**: uma CPU híbrida aparece como não híbrida e a contagem de núcleos sai errada. Quando o modelo
da CPU sugere que ela é híbrida e isso não foi detectado, o `ovbench` mostra um aviso. Duas saídas:

- Rodar no Windows ou no Linux nativos (a detecção funciona).
- Informar a topologia à mão com `--cpu-topology` (ou a variável `OVBENCH_CPU_TOPOLOGY`):

```bash
ovbench info --cpu-topology 8P+4E          # 8 P-cores com HT + 4 E-cores (ex.: Core i7-12700)
ovbench run  --cpu-topology 8P+4E --model modelos/qwen25-0.5b
```

Formatos: `8P+4E` (P-cores com Hyper-Threading), `8P+4E,noht` (sem HT) e `6P` (sem E-cores). O relatório
registra se a topologia foi `detectada` ou `informada`.

## Presets

Os presets são gerados da topologia detectada. `SCHEDULING_CORE_TYPE` só aparece em CPU híbrida.

| CPU | Presets |
|---|---|
| Híbrida (Core 12ª geração em diante, Core Ultra) | `pcore_latency`, `pcore_sem_ht`, `pcore_kv_cache_u8`, `pcore_throughput`, `ecore_throughput`, `todos_os_nucleos` |
| Não híbrida (Xeon, Core até a 11ª geração) | `latency_fisicos`, `latency_logicos`, `kv_cache_u8`, `throughput` |
| Qualquer CPU | `default` (a escolha do OpenVINO), `sem_pinagem` |
| `--device GPU`, `GPU.0`, `NPU` | `default`, `throughput` |

Para ver a lista da sua máquina: `ovbench run --list-presets`. Para testar uma configuração própria:

```bash
ovbench run --model modelos/qwen25-0.5b --ov-config 'meu={"INFERENCE_NUM_THREADS":"6"}' --presets meu
```

Se o OpenVINO rejeitar uma configuração, ela aparece como `erro` no relatório e as demais continuam.

## Modos de execução

**Instância** (padrão): carrega uma instância por preset e roda os prompts em sequência. Isola o efeito de
cada configuração, sem ruído de concorrência.

**Pool**: carrega N instâncias e dispara requisições concorrentes, medindo a espera em fila e o throughput.

```bash
ovbench run --model modelos/qwen25-0.5b --mode pool --pool-instancias 2 --pool-concorrencia 20 --repetitions 5
```

## Medições estáveis

Notebooks e desktops reduzem o clock quando esquentam. Se os presets rodam em sequência, o último sofre com
isso e a comparação sai enviesada. Para evitar:

| Opção | Efeito |
|---|---|
| `--interleave` | Intercala as repetições entre os presets, com a ordem de cada rodada em desenho balanceado (cada preset passa por cada posição o mesmo número de vezes; use um nº de repetições múltiplo do nº de presets). Todos os presets ficam carregados ao mesmo tempo, então a RAM é a soma dos modelos. |
| `--shuffle` | Embaralha a ordem dos presets no modo sequencial. |
| `--cooldown S` | Pausa de S segundos entre blocos, para a CPU esfriar. |
| `--seed N` | Torna a ordem reproduzível (padrão 42). |

Boas práticas: deixe o notebook na tomada em modo de alto desempenho, feche outros programas e use 6 ou mais
repetições antes de tirar conclusões de diferenças pequenas.

```bash
ovbench run --model modelos/qwen25-0.5b --presets default latency_fisicos kv_cache_u8 \
  --repetitions 8 --fixed-length 128 --interleave --cooldown 5
```

## Comparar modelos de forma justa

Modelos diferentes geram respostas de tamanhos diferentes, e a latência acaba refletindo o tamanho da resposta
e não a velocidade. Para comparar modelos:

- Use `--fixed-length N`: todos geram exatamente N tokens.
- Use `--input-lengths 128 512 2048` para ver como o prefill escala com o tamanho do prompt.
- Use o mesmo `--workload`, o mesmo preset e as mesmas repetições em cada modelo.
- Para modelos híbridos de raciocínio (por exemplo Qwen3), passe `--enable-thinking false`.

## Cargas de teste e validadores

**Workload** (`--workload`): `geral` e `curto` (embutidos) ou um arquivo.

```jsonl
{"id": "extracao", "system": "Responda somente com JSON.", "prompt": "Extraia nome e cidade: 'Maria mora em Curitiba.'"}
{"id": "chat", "messages": [{"role": "user", "content": "Explique o que é uma CPU híbrida."}]}
```

**Validador** (`--validator`): `none`, `json`, `nonempty`, `regex:PADRÃO`, `pacote.modulo:funcao` ou
`arquivo.py:funcao`. Uma função de validação recebe o texto da resposta e devolve `True` ou `False`.

## Referência do `ovbench run`

| Opção | Padrão | Descrição |
|---|---|---|
| `--model` | | Pasta do modelo em OpenVINO IR |
| `--device` | `CPU` | `CPU`, `GPU`, `GPU.0`, `NPU` |
| `--cpu-topology` | detectada | Informa a topologia à mão, ex.: `8P+4E` (veja "Topologia da CPU") |
| `--mode` | `instancia` | `instancia` ou `pool` |
| `--presets` | todos | Presets a testar (no pool, um preset é o baseline) |
| `--ov-config` | | `NOME=JSON` ou arquivo `.json`; repetível |
| `--list-presets` | | Lista os presets e sai |
| `--workload` | `geral` | Nome embutido ou arquivo |
| `--input-lengths` | | Acrescenta prompts de ~N tokens de entrada |
| `--repetitions` | 3 | Repetições do conjunto de prompts |
| `--warmup` | 1 | Gerações de aquecimento, fora da medição |
| `--max-new-tokens` | 256 | Limite de tokens gerados |
| `--fixed-length` | | Gera exatamente N tokens |
| `--validator` | `none` | Ver a seção de validadores |
| `--enable-thinking` | | `true` ou `false`, para modelos Qwen3 |
| `--trust-remote-code` | | Para modelos que exigem código remoto |
| `--interleave`, `--ordem`, `--shuffle`, `--cooldown`, `--seed` | | Ver a seção de medições estáveis |
| `--pool-instancias`, `--pool-concorrencia` | 2, nº de instâncias | Modo pool |
| `--save-responses` | | Grava as respostas completas em `.jsonl` |
| `--out` | `resultados` | Pasta de saída |

## Uso como biblioteca

```python
from ovbench.backends import make_spec
from ovbench.hardware import detect_topology
from ovbench.presets import build_presets
from ovbench.runner import RunConfig, run_instance
from ovbench.workloads import load_workload

spec = make_spec("openvino", "modelos/qwen25-0.5b")
presets = build_presets(detect_topology())
r = run_instance(spec.factory, "default", presets["default"],
                 load_workload("geral"), RunConfig(repetitions=3, fixed_length=128))
print(r["resumo"]["decode_tok_s"]["mean"], "tok/s")
```

## Estrutura

```
src/ovbench/
  backends/      contrato LLMBackend, backend OpenVINO e backend fake (testes)
  hardware.py    detecção de topologia, instruções e ambiente
  presets.py     geração de configurações a partir da topologia
  workloads.py   prompts embutidos, arquivos e escada de tamanhos
  validators.py  validadores de formato
  metrics.py     estatísticas, TTFT, tok/s e pico de RAM
  runner.py      modos instância, intercalado e pool
  report.py      relatórios JSON, CSV e Markdown
  convert.py     conversão e quantização via optimum-cli
  download.py    download de modelos já em OpenVINO IR (Hugging Face)
  cli.py         ovbench info | run | convert | download
tests/           testes com backend falso (não exigem modelo nem OpenVINO)
```

## Limitações

**O que os números significam**
- Os tokens/s servem para **comparar modelos e configurações na mesma máquina**, não como valor absoluto do
  hardware. A geração passa por `optimum-intel` e PyTorch, o que adiciona um custo de Python que não existe
  num pipeline nativo do OpenVINO. Esse custo é igual em todos os testes, então a comparação continua válida,
  mas os valores não são comparáveis com tabelas publicadas por outras ferramentas.
- O `ovbench` não substitui o `benchmark_app` do OpenVINO. O `benchmark_app` mede uma inferência de rede com
  entradas sintéticas; o `ovbench` mede uma geração de texto completa (TTFT e decodificação). Eles respondem a
  perguntas diferentes.
- Só velocidade e uso de memória. Não há nota de qualidade das respostas: use `--save-responses` para ler as
  respostas e `--validator` para checar o formato.
- Não mede a latência de cada token, apenas o TTFT e a média da decodificação.
- Diferenças pequenas (poucos por cento) ficam dentro do ruído, sobretudo em notebooks. Use repetições
  suficientes e `--interleave` antes de concluir algo.

**Escopo**
- Só OpenVINO. Outros backends (llama.cpp, Ollama) estão fora do escopo.
- O modelo precisa estar em formato OpenVINO IR (o `ovbench convert` ajuda a converter).
- A geração é gulosa (sem amostragem), para tornar a medição determinística.
- Os presets de iGPU e NPU são mínimos. Em máquinas com GPU de outro fabricante, escolha o dispositivo Intel
  com `--device GPU.N` (veja `ovbench info`).
- No modo `--interleave`, todos os presets ficam na memória ao mesmo tempo.

**Detecção de hardware**
- Em WSL2 e máquinas virtuais a topologia real da CPU fica escondida; use `--cpu-topology` (veja a seção
  "Topologia da CPU").
- A detecção de núcleos P/E usa `GetSystemCpuSetInformation` no Windows e `/sys/devices` no Linux. Em macOS a
  CPU é tratada como não híbrida.
- Em CPU híbrida, a contagem de P-cores físicos no Linux depende do `core_id` do sistema.
- As instruções (AVX2, AVX-512, VNNI, AMX) vêm do `/proc/cpuinfo` no Linux e do `py-cpuinfo` no Windows. Se o
  `py-cpuinfo` falhar, só `avx2` e `avx512f` são detectadas. No Windows, o `avx_vnni` não aparece mesmo com o
  `py-cpuinfo` instalado (visto numa i7-12700, onde o Linux/WSL o reporta), então uma lista incompleta de
  instruções no Windows não significa que a CPU não tenha o recurso. O campo `isa` é só informativo: não
  influencia presets nem medições.

**Testado**
- Core i7-1165G7 (4 núcleos / 8 threads, não híbrido), Windows 11, Python 3.12, OpenVINO 2026.4,
  optimum-intel 2.2 e transformers 5.5, com Qwen2.5-0.5B em INT8.
- Core i7-12700 (8P + 4E, 20 threads), Windows 11 **nativo**, Python 3.13, mesmas versões, com
  `OpenVINO/Qwen2.5-1.5B-Instruct-int8-ov`: a topologia 8P+4E foi detectada automaticamente (sem avisos) e
  os presets de P-core/E-core foram gerados. Nessa máquina a escolha de núcleos pesou (veja abaixo).
- Mesma CPU em **WSL2**, com `--cpu-topology 8P+4E`: todos os presets ficaram praticamente iguais
  (~26 tok/s, diferença menor que 4%). No Windows nativo, os mesmos presets se separaram em até 24%. Ou seja,
  em WSL2/VM a pinagem de núcleos não tem o efeito esperado: não tire conclusões sobre núcleos de uma medição
  feita aí.
- Os presets de GPU e NPU só foram cobertos por testes automatizados.

**Exemplo em CPU híbrida** (i7-12700, Windows 11 nativo, Qwen2.5-1.5B INT8, 6 repetições, 128 tokens):

| Preset | Decode (tok/s) | Lat. média (ms) | TTFT (ms) |
|---|---|---|---|
| pcore_kv_cache_u8 | 24,5 | 5341 | 146 |
| pcore_latency | 23,0 | 5731 | 159 |
| pcore_sem_ht | 22,6 | 5947 | 168 |
| todos_os_nucleos | 22,3 | 6103 | 311 |
| default | 19,7 | 6677 | 215 |
| sem_pinagem | 18,7 | 7108 | 298 |

Nessa rodada a ordem dos presets em cada rodada era aleatória e o `pcore_kv_cache_u8` caiu sempre nas
posições finais; por isso o desenho passou a ser balanceado por padrão. Trate a vantagem dele sobre o
`pcore_latency` como indicação, não como conclusão.

## Testes

```bash
pytest
```

Os testes usam um backend falso e não precisam de modelo.

## Licença

MIT. Veja [LICENSE](LICENSE).
