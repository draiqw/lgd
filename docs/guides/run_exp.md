## 1. Сборка Docker образа

### Для Linux/Intel Mac:

```bash
docker build --load -t llabs_lda_hyperopt .
```

### Для Apple Silicon (M1/M2/M3):

```bash
docker buildx build --platform linux/amd64 --load -t llabs_lda_hyperopt .
```

### Проверка образа (опционально):

```bash
docker images | grep llabs_lda_hyperopt
```

---

## 2. Публикация в Docker Hub

### Тегирование и публикация:

Замените `YOUR_DOCKERHUB_USERNAME` на ваш username:

```bash
docker tag llabs_lda_hyperopt YOUR_DOCKERHUB_USERNAME/llabs_lda_hyperopt:latest
```
```bash
docker push YOUR_DOCKERHUB_USERNAME/llabs_lda_hyperopt:latest
```
Пример
```bash
docker tag llabs_lda_hyperopt draiqws/llabs_lda_hyperopt:latest
```
```bash
docker push draiqws/llabs_lda_hyperopt:latest
```


---

## 4. Конвертация образа через enroot

### Подключение к виртуалке с enroot:

```bash
ssh -p 2295 mmp@188.44.41.125
```

### Конвертация Docker образа в .sqsh:

```bash
sudo enroot import docker://YOUR_DOCKERHUB_USERNAME/llabs_lda_hyperopt:latest
```

**Пример:**
```bash
sudo enroot import docker://draiqws/llabs_lda_hyperopt:latest
```

После завершения enroot выведет путь к `.sqsh` файлу:

```
[INFO] Importing Docker image
...
[INFO] Image saved to: /home/mmp/YOUR_DOCKERHUB_USERNAME+llabs_lda_hyperopt+v1.sqsh
```

Запомните этот путь!

---

## 5. Загрузка на slurm-master

### Скачать .sqsh с виртуалки на локальный компьютер:

Откройте **новый терминал** на вашем компьютере:

```bash
scp -P 2295 mmp@188.44.41.125:/home/mmp/akramovrr/draiqws+llabs_lda_hyperopt+latest.sqsh .
```

### Загрузить на slurm-master:

Замените `YOUR_USERNAME` на ваш username на slurm-master:

```bash
scp draiqws+llabs_lda_hyperopt+latest.sqsh akramovrr@10.36.60.202:/scratch/akramovrr/
```


### Загрузить run.sh на slurm-master:

```bash
scp run.sh akramovrr@10.36.60.202:/scratch/akramovrr/
```

---

## 6. Запуск эксперимента

### Подключение к slurm-master:

```bash
ssh akramovrr@10.36.60.202
```

### Использование tmux (рекомендуется):

Если планируется отключаться от сессии, используйте `tmux`:

```bash
tmux new -s lda_experiment

# Отключиться от сессии (не завершая её): Ctrl+b, затем d
# Вернуться к сессии позже:
tmux a -t lda_experiment
```

### Перейти в рабочую директорию:

```bash
cd /scratch/$USER
```

### Запустить задачу через sbatch:

```bash
sbatch run.sh
```

Вы получите сообщение:
```
Submitted batch job 12345
```

Где `12345` - это ID вашей задачи.

---

## 7. Мониторинг и получение результатов

### Проверить статус задачи:

```bash
squeue -u $USER
```

**Расшифровка статусов:**
- `PD` (Pending) - задача в очереди, ожидает ресурсов
- `R` (Running) - задача выполняется
- `CG` (Completing) - задача завершается
- Если задачи нет в списке - она завершилась

### Просмотр логов в реальном времени:

```bash
tail -f slurm-12345.out
```

```bash
tail -f slurm-12345.err
```

### Мониторинг через Grafana:

Откройте в браузере:
```
http://10.36.60.3:3000/
```

### Просмотр результатов:

```bash
/scratch/akramovrr/lda_results/
```

Проверить структуру результатов:
```bash
ls -lh /scratch/akramovrr/lda_results/
```

### Скачать результаты на локальный компьютер:

Откройте терминал на **вашем компьютере**:

```bash
scp -r akramovrr@10.36.60.202:/scratch/akramovrr/lda_results/ ./local_results/
```

---

## 8. Полезные команды SLURM

### Просмотр очереди задач:

```bash
# Все задачи в очереди
squeue
```
```bash
# Только ваши задачи
squeue -u $USER
```
```bash
# Детальная информация о задаче
scontrol show job 12345
```
```bash
# Отменить конкретную задачу
scancel 12345
```
```bash
# Отменить все ваши задачи
scancel -u $USER
```