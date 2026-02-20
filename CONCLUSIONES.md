# Conclusiones del Analisis ETL
## Base de Datos de Empresas Activas - Camara de Comercio de Ibague

**Autor:** Juan Camilo Perea Possos
**Fecha:** 20 de Febrero de 2026

---

### 1. El tejido empresarial de Ibague esta dominado por empresas muy jovenes

La antiguedad promedio de las empresas registradas es de apenas **5.5 anos**, con una mediana de **0.8 anos**. Esto significa que la mitad de las empresas activas tienen menos de un ano de haberse matriculado. El 75% de las empresas tiene **1.2 anos o menos** de antiguedad, mientras que la empresa mas antigua alcanza los 54.5 anos. Este patron evidencia una alta dinamica de creacion empresarial reciente en la region, pero tambien sugiere una posible alta tasa de mortalidad empresarial, ya que las empresas con mayor antiguedad representan una minoria significativa del total de 20,280 registros.

---

### 2. El comercio es el sector economico predominante en la region

El analisis de las secciones economicas CIIU revela que el sector **G (Comercio al por mayor y al por menor)** concentra **7,578 empresas (37.4%)**, siendo por amplio margen la actividad mas frecuente. Le siguen el sector **I (Alojamiento y servicios de comida)** con 3,447 empresas (17%) y el sector **S (Otras actividades de servicios)** con 3,115 empresas (15.4%). La manufactura (sector C) representa solo el 6.3% con 1,277 empresas. Esta distribucion indica que la economia de Ibague se sustenta principalmente en actividades comerciales y de servicios, con una participacion industrial relativamente baja en comparacion.

---

### 3. La calidad de los datos originales presenta desafios significativos

Del total de 1,723,800 celdas en el dataset original (20,280 filas x 85 columnas), el **44.47% resultaron ser valores nulos** despues de reemplazar los marcadores "No reporta" y "No aplica". Se identificaron **27 columnas con mas del 80% de datos faltantes**, incluyendo variables como EMAIL COMERCIAL 2 y 3 (100% nulos), IMPORTA/EXPORTA (99.7%), TIEMPO FUNCIONAMIENTO (99.4%) y multiples fechas de pago de renovacion historicas (2016-2023). Esto redujo el dataset de 85 a 58 columnas utiles. La alta proporcion de datos ausentes refleja que muchas variables del registro mercantil no son diligenciadas de manera obligatoria o consistente por los comerciantes al momento de su inscripcion.

---

### 4. Las microempresas conforman la base del ecosistema empresarial

Del analisis de tamano empresarial se evidencia que las **microempresas** representan la gran mayoria del tejido empresarial de Ibague. El 45.1% de los registros no reportan clasificacion de tamano (corresponden principalmente a establecimientos de comercio), pero de los que si la reportan, la categoria de micro empresa es abrumadoramente dominante. Esto se refleja tambien en que el **Grupo NIIF III (Microempresas)** es el mas frecuente entre las empresas clasificadas. Este hallazgo es coherente con la estructura economica colombiana, donde las MiPymes representan mas del 90% del tejido empresarial nacional.

---

### 5. La participacion femenina en el liderazgo empresarial es limitada

Los datos muestran que la mediana de **cantidad de mujeres por empresa es 0**, lo que indica que mas de la mitad de las empresas registradas no reportan mujeres vinculadas. La participacion de mujeres en cargos directivos sigue un patron similar con valores predominantemente bajos. El porcentaje de participacion de mujeres se concentra en los extremos (0% o valores muy bajos), con casos puntuales de participacion alta. Este hallazgo evidencia una brecha de genero en el ecosistema empresarial de Ibague, lo que podria orientar politicas publicas de fomento al emprendimiento femenino y la inclusion de mujeres en posiciones de liderazgo empresarial en la region.
