# Estrategia de Seguridad, Cumplimiento Legal y Acceso por Roles

## 1. Cumplimiento Ley N° 19.628 (Proteccion de la Vida Privada, Chile)

### 1.1. Contexto Legal

La Ley N° 19.628 sobre Proteccion de la Vida Privada, vigente en Chile desde 1999,
regula el tratamiento de datos personales y establece las obligaciones de los
responsables de los datos. Esta normativa exige que los datos sensibles (como
numeros de tarjeta de credito, nombres, direcciones) sean protegidos durante su
almacenamiento, procesamiento y transmision.

### 1.2. Implementacion Tecnica: Pseudoanonimizacion via SHA-256

En la etapa `src/02_cleaning.py` se aplica un enmascaramiento irreversible
mediante la funcion hash SHA-256 sobre las siguientes columnas identificadas
como datos personales sensibles:

| Columna | Dato protegido | Riesgo asociado |
|---------|---------------|-----------------|
| `cc_num` | Numero de tarjeta de credito | Robo financiero, clonacion |
| `first` | Nombre del titular | Identificacion directa |
| `last` | Apellido del titular | Identificacion directa |
| `street` | Direccion particular | Geolocalizacion residencial |

El proceso sigue tres principios tecnicos:

1. **Irreversibilidad**: SHA-256 es una funcion hash criptografica unidireccional.
   Dado un hash `h = SHA-256(x)`, no existe un algoritmo eficiente para recuperar
   `x`. Esto cumple con el Art. 7° de la Ley 19.628 que exige que los datos
   sensibles sean almacenados de forma que "no puedan asociarse al titular".

2. **Determinismo controlado**: Una misma entrada siempre produce el mismo hash,
   lo que permite operaciones de integridad referencial (ej. deteccion de
   transacciones duplicadas del mismo `cc_num`) sin revelar el numero real.

3. **Cumplimiento en transferencias**: Segun el Art. 5° de la Ley 19.628, la
   transferencia de datos personales requiere autorizacion del titular. Al
   aplicar pseudoanonimizacion antes del almacenamiento en parquet/SQLite, los
   datos almacenados en `data/processed/` ya no contienen informacion personal
   directa, eliminando la necesidad de consentimiento explicito para su
   transferencia dentro del ambiente DataOps.

### 1.3. Diagrama de Flujo de Proteccion de Datos

```
[Dato Crudo] --> SHA-256(hash) --> [hash almacenado]
  cc_num=1234       =/=>            e3b0c442...
  first=Juan        =/=>            a7ffc6f4...
  last=Perez        =/=>            3fdba35f...

  Reversibilidad: IMPOSIBLE (colision preimagen computacionalmente inviable)
  Integridad:     POSIBLE  (mismo hash = mismo original)
```

### 1.4. Analisis ante la Agencia Reguladora

La estrategia implementada se alinea con las recomendaciones de la Agencia de
Proteccion de Datos Personales (Chile) y el Reglamento General de Proteccion de
Datos (GDPR, Union Europea) como estandar internacional:

- **Pseudoanonimizacion vs Anonimizacion**: El hash SHA-256 constituye una
  pseudoanonimizacion, ya que teoricamente existe una correspondencia
  uno-a-uno con el dato original. Sin embargo, al no almacenarse la clave de
  desencriptacion en ningun punto del sistema, el riesgo de reidentificacion
  se reduce drasticamente.
- **Almacenamiento minimo**: Solo se conservan las columnas estrictamente
  necesarias para el modelamiento. Las columnas `trans_num`, `unix_time`, `dob`
  se descartan en la etapa de feature engineering (`src/05_model_training.py`,
  `COLUMNAS_A_DESCARTAR`).
- **Segregacion de datos**: Los datos crudos (`data/raw/`) estan en el
  `.gitignore` y no se distribuyen en imagenes Docker ni en el repositorio.
  Solo los datos procesados viajan al contenedor de produccion.



## 2. Control de Acceso Basado en Roles (RBAC)

### 2.1. Matriz de Roles

Se definen tres perfiles con privilegios estrictamente limitados segun el
principio de *minimum necessary access* (acceso minimo necesario):

| Recurso | Data Engineer / Admin | Data Scientist | Analista de Negocio |
|---------|----------------------|---------------|-------------------|
| **Docker Compose / Infraestructura** | Lectura/Escritura | Sin acceso | Sin acceso |
| **Scripts raiz (src/)** | Lectura/Escritura | Lectura | Sin acceso |
| **Hiperparametros (config.py)** | Lectura/Escritura | Lectura/Escritura | Sin acceso |
| **Datos crudos (data/raw/)** | Lectura/Escritura | Sin acceso | Sin acceso |
| **Datos procesados (data/processed/)** | Lectura/Escritura | Lectura | Sin acceso |
| **Modelo entrenado (models/)** | Lectura/Escritura | Lectura/Escritura | Sin acceso |
| **Logs de pipeline (logs/)** | Lectura/Escritura | Lectura | Sin acceso |
| **Dashboard Metabase** | Lectura/Escritura | Lectura | Solo lectura |
| **Contenedores Docker (shell)** | Acceso total | Sin acceso | Sin acceso |
| **Archivos del sistema host** | Sin acceso | Sin acceso | Sin acceso |

### 2.2. Implementacion en la Arquitectura

La segmentacion de accesos se materializa mediante:

- **Volumenes Docker**: Cada servicio monta solo los directorios que necesita.
  - `pipeline`: monta `data/`, `models/`, `logs/`, `reports/`
  - `streamlit`: monta `data/` (read), `models/` (read)
  - `metabase`: monta `data/` como solo lectura (`:ro`)
- **Perfil de servicios**: El servicio `pipeline-telemetry` se ejecuta bajo un
  perfil separado (`--profile telemetry`) accesible solo por administradores.
- **Red interna**: Los servicios streamlit y metabase dependen de la finalizacion
  exitosa del pipeline, pero no comparten redes abiertas entre si.

### 2.3. Privacidad por Diseno

- El Analista de Negocio **nunca** tiene acceso a los datos crudos ni a los
  scripts del pipeline.
- Las credenciales de Metabase se gestionan internamente (sin exponerlas en
  docker-compose).
- Los logs del pipeline no contienen datos personales gracias al enmascaramiento
  SHA-256 aguas arriba.



## 3. Limitaciones Operacionales del Sistema Actual

### 3.1. SQLite y Concurrencia

SQLite es un motor de base de datos embebido que utiliza bloqueo a nivel de
archivo mediante locks exclusivos en escritura. Esto lo hace inadecuado para
escenarios de alta concurrencia donde multiples procesos o usuarios necesiten
escribir simultaneamente. En la arquitectura actual:

- Metabase intenta consultas de solo lectura sobre `fraud_detection.db`
- El pipeline escribe secuencialmente (una etapa a la vez)
- **Riesgo**: Si dos procesos intentaran escribir en paralelo (ej. reentrenamiento
  mientras se carga un nuevo batch), SQLite lanzaria un error
  `database is locked`. La base de datos no escala horizontalmente.

### 3.2. Desbalance de Clases Extremo (603:1)

Con 603 transacciones legitimas por cada fraude en el conjunto de test, el
modelo opera en un regimen de desbalance severo. Esto produce:

- **Precision baja**: Solo 1 de cada 4 alertas (umbral 0.5) corresponde a un
  fraude real. Para el umbral de negocio (0.25), solo 1 de cada 8.
- **F1-Score limitado**: La media armonica entre precision y recall alcanza solo
  0.3652 en el umbral default, lo que refleja el trade-off forzado por la
  distribucion de clases.
- **Calibracion de probabilidades**: Las probabilidades predichas por XGBoost
  no estan calibradas; el umbral de 0.25 se eligio heuristicamente, no mediante
  optimizacion sobre la curva Precision-Recall.

### 3.3. Riesgo de Sobregjuste por SMOTE con k_neighbors=3

SMOTE genera muestras sinteticas interpolando entre vecinos cercanos de la
clase minoritaria. Con `k_neighbors=3` y solo 1.961 casos de fraude originales:

- **Sinteticos de baja calidad**: La interpolacion entre 3 vecinos en un espacio
  de 17 dimensiones puede generar puntos que no representan patrones de fraude
  reales, especialmente si los 3 vecinos son muy cercanos entre si.
- **Generalizacion comprometida**: El modelo puede aprender patrones especificos
  de los sinteticos que no existen en la poblacion real de fraudes.
- **Validacion insuficiente**: Con solo 184 fraudes en test (0.03% del dataset),
  las metricas de recall/F1 tienen intervalos de confianza amplios.

### 3.4. Falta de Orquestador Dinamico

El pipeline actual se ejecuta mediante un comando secuencial en Docker Compose:

```bash
python src/01_ingestion.py && python src/02_cleaning.py && ...
```

Esto presenta las siguientes limitaciones frente a un orquestador como Apache
Airflow, Prefect o Dagster:

| Capacidad | Script secuencial | Orquestador (Airflow) |
|-----------|------------------|----------------------|
| Reintentos automaticos | No | Si (con backoff) |
| Notificaciones en fallo | No | Si (email, Slack) |
| Paralelismo de stages | No | Si (topologia DAG) |
| Reprocesamiento parcial | Manual | Si (backfill) |
| Monitoreo de SLA | No | Si (alertas por retraso) |
| Versionado de ejecuciones | No | Si (DAG runs) |
| Lineage de datos | No | Si (con extensiones) |

### 3.5. Otras Limitaciones Identificadas

- **Sin monitoreo de deriva**: No hay deteccion de data drift ni concept drift
  una vez el modelo esta en produccion. Las metricas reportadas son estaticas
  del momento de entrenamiento.
- **Sin versionado de datos**: No se utiliza DVC ni LakeFS para mantener
  lineage sobre las versiones del dataset. Un cambio en la fuente cruda no es
  trazable.
- **Dependencia de uptime de Docker**: El servicio streamlit depende del estado
  del demonio Docker. Si el contenedor cae, no hay auto-recuperacion sin
  intervencion manual (a menos que se use `restart: unless-stopped`).
- **Escalabilidad vertical**: La imagen Docker no esta optimizada para
  procesamiento distribuido. El pipeline entero corre en un solo contenedor.



## 4. Recomendaciones para Produccion

1. **Reemplazar SQLite por PostgreSQL** para soporte de concurrencia real y
   conexiones simultaneas desde Metabase y servicios externos.
2. **Implementar Airflow o Prefect** para orquestar el pipeline con reintentos,
   notificaciones y monitoreo de SLA.
3. **Agregar calibracion de probabilidades** via Platt Scaling o Isotonic
   Regression para mejorar la precision de las predicciones.
4. **Evaluar ensemble con Isolation Forest** como capa de deteccion de anomalias
   no supervisada complementaria a XGBoost.
5. **Implementar Whylogs o Evidently** para monitoreo continuo de deriva de
   datos y concepto.
6. **Configurar restart policies** en Docker Compose para resiliencia ante caidas
   de contenedores.



*Documento generado para la Evaluacion Parcial 2 - Gestion de Datos para IA (Duoc UC)*
*Ultima actualizacion: junio 2026*
