from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from blog.blog_users.services import UserService


@api_view(['GET'])
@parser_classes([JSONParser])
def login(request):
    users = UserService().create_users()
    return JsonResponse({'users ': users})


@api_view(['GET'])
@parser_classes([JSONParser])
def user_list(request):
    service = UserService()
    service.get_users()
    return JsonResponse({"result":service.get_users()})


'''
    user_info = request.data
    email = user_info['email']
    password = user_info['password']
    print(f'리액트에서 보낸 데이터: {user_info}')
    print(f'넘어온 이메일 : {email}')
    print(f'넘어온 비밀번호: {password}')'''