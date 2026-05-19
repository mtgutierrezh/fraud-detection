## Carta Gantt (PMBOK/WBS)

Cronograma del proyecto con metodologia hibrida PMBOK/Agile (17 tareas atomicas WBS).

```mermaid
gantt
    title Carta Gantt - Fraud Detection DataOps
    dateFormat YYYY-MM-DD
    axisFormat %d-%b

    section Gestion y Planificacion
    Justificar metodologia PMBOK/Agile     :done,    t1, 2026-05-13, 2d
    Crear Carta Gantt / WBS                :done,    t2, 2026-05-14, 1d
    Definir roles DataOps del equipo       :done,    t3, 2026-05-14, 1d

    section Infraestructura y Seguridad
    Configurar Dockerfile y requirements   :done,    t4, 2026-05-14, 2d
    Estructurar repositorio y .gitignore   :done,    t5, 2026-05-14, 1d

    section Pipeline de Datos Core
    Script 01 - Ingesta de datos           :done,    t6, 2026-05-15, 1d
    Script 02 - Enmascaramiento PII        :done,    t7, after t6, 1d
    Script 02 - Limpieza e imputacion      :done,    t8, after t6, 1d
    Script 03 - Validacion Estructural     :done,    t9, after t8, 1d
    Script 04 - Carga de datos finales     :done,    t10, after t9, 1d

    section Monitoreo y Cierre
    Configurar Logs centralizados          :done,    t11, 2026-05-18, 1d
    Definir KPIs y Plan de Escalabilidad   :active,  t12, 2026-05-19, 1d

    section Modelamiento y Despliegue
    Script 05 - Entrenar XGBoost           :active,  t15, 2026-05-20, 1d
    Desarrollar app Streamlit              :         t16, 2026-05-21, 1d
    Desplegar en Render y docs             :         t17, 2026-05-21, 1d

    section Preparacion Final
    Ensamblar Informe Tecnico 10-12 pag    :         t13, 2026-05-19, 2d
    Preparar y ensayar Demo en vivo        :         t14, 2026-05-21, 1d

    section Hitos
    Entrega de entregables                 :milestone, m1, 2026-05-20, 0d
    Defensa oral 15 min                    :milestone, m2, 2026-05-22, 0d
```

## Hitos clave

| Fecha | Hito |
|-------|------|
| 2026-05-13 | Inicio del proyecto |
| 2026-05-20 | Entrega de entregables (informe + codigo + Docker + modelo) |
| 2026-05-22 | Defensa oral (15 min, 3 integrantes) |
