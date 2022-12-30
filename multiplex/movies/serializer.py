from rest_framework import serializers
from multiplex.movies.models import Movie


class MovieSerializer(serializers.ModelSerializer):
    class Meta:
        model = Movie
        fields = '__all__'

    def create(self,validated_data):
        return Movie.objects.create(**validated_data)

    def update(self,instace,valicated_data):
        Movie.objects.filter(pk=instace.id).update(**valicated_data)

    def delete(self,instance,valicated_data):
        pass

