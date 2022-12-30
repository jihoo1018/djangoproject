from django.http import JsonResponse
from rest_framework import status
from rest_framework.parsers import JSONParser
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response

from blog.blog_users.serializer import BlogUserSerializer
from blog.blog_users.repositories import UserRepository


'''@api_view(['GET'])
@parser_classes([JSONParser])
def login(request):
    users = UserService().create_users()
    return JsonResponse({'users ': users})'''


@api_view(['POST','PUT','PATCH','DELETE','GET'])
@parser_classes([JSONParser])
def blog_user_view(request):
    if request.method == "POST":
        new_user = request.data
        print(f"리액트에서 등록한 신규 사용자 {new_user}")
        serializer = BlogUserSerializer(data=new_user)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse({"result":"success"})
        return JsonResponse(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    elif request.method == "PATCH":
        return None
    elif request.method == "PUT":
        rep = UserRepository()
        modify_user = rep.find_by_email(request.data['email'])
        db_user = rep.find_by_id(modify_user.blog_userid)
        serializer = BlogUserSerializer(data=db_user)
        if serializer.is_valid():
            serializer.update(modify_user,db_user)
            return JsonResponse({"result":"success"})
    elif request.method == "GET":
        return Response(UserRepository().find_by_email(request.data['email']))
    elif request.method == "DELETE":
        rep = UserRepository()
        delete_user = rep.find_by_email(request.data['email'])
        db_user = rep.find_by_id(delete_user.blog_userid)
        db_user.delete()
        return JsonResponse({"result":"success"})

@api_view(['GET'])
@parser_classes([JSONParser])
def user_list(request):
    return UserRepository().get_all()

@api_view(['GET'])
@parser_classes([JSONParser])
def user_list_name(request):
    return UserRepository().find_by_name(request.data['nickname'])

@api_view(['POST'])
def loginform(request): return UserRepository().login(request.data)
