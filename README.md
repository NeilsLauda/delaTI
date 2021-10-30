# t-ml-backend
#Link del backend en producción heroku: https://js17-ml-backend.herokuapp.com/
_______
#Link del repositorio frontend: https://github.com/Julisa2020/js17-ml-frontend
_______
#Endpoints:
_____________
> get Algorithms :  METHOD: GET https://js17-ml-backend.herokuapp.com/algorithms
_____________
> kmeans : METHOD POST https://js17-ml-backend.herokuapp.com/kmeans
_____________
>QUERY: 
```
2 COLUMNS:
          {
    "query":"select distinct o.htitulo_cat, o.htitulo from webscraping w inner join oferta o on (w.id_webscraping=o.id_webscraping) where o.id_estado is null order by 1,2 limit 500;",
    "n_clusters": 5,
    "init": "random",
    "max_iter": 500
          }
14 COLUMNS:
{
    "query":"select o.htitulo_cat as categoria,o.htitulo as perfil, w.pagina_web,o.empresa,o.lugar,o.salario,date_part('year',o.fecha_publicacion) as periodo, f_dimPerfilOferta(o.id_oferta,7) as funciones, f_dimPerfilOferta(o.id_oferta,1) as conocimiento, f_dimPerfilOferta(o.id_oferta,3) as habilidades, f_dimPerfilOferta(o.id_oferta,2) as competencias, f_dimPerfilOferta(o.id_oferta,17) as certificaciones, f_dimPuestoEmpleo(o.id_oferta,5) as beneficio, f_dimPerfilOferta(o.id_oferta,11) as formacion from webscraping w inner join oferta o on (w.id_webscraping=o.id_webscraping) where o.id_estado is null limit 500;",
    "n_clusters": 5,
    "init": "random",
    "max_iter": 500
      }
```
