import fb
import facebook as fbsdk
import requests
import json
import csv


token="EAACEdEose0cBAEHhIcBN4rt4YG6ibFVvZBt2NsJG6uj6u7eHNX3hNUTk6pA5I7ULZCsQtxWcgvbgN2Fx6zYPIjDKHAlwIDyFAN23a7BSlIy2SIYXruNnRJ92PYhUChime9wqE9f9ZAf1R8HLZBmgFZCToLPOoyR345kw9UiMfYbEGknDeZANHplWfPWSpmNeZARWcWefZBxAYQZDZD"


def get_myfacebook_likes(myfacebook_graph,id):
   myfacebook_likes = []
   myfacebook_likes_info = myfacebook_graph.get_connections(id, "likes")

   while myfacebook_likes_info['data']:
     for like in myfacebook_likes_info['data']:
       myfacebook_likes.append(like)
     if 'next' in myfacebook_likes_info['paging'].keys():
       myfacebook_likes_info = requests.get(myfacebook_likes_info['paging']['next']).json()
     else:
       break

   return myfacebook_likes


def get_likes_on_post(id):
    count=0
    url="https://graph.facebook.com/v2.10/"+id+"/reactions?access_token="+token
    while(True):
        r = requests.get(url)
        r=r.json()
        #print(r)
        #for i in range(len(r['data'])):
            #names=r['data'][i]['name']
            #print(names)
        if('data' in r):
            count=count+len(r['data'])
        else:
            break
        if not 'paging' in r:
            break
        else:
            if not 'next' in r['paging']:
                break
        url=r['paging']['next']
    return count


def get_reactions_on_post(id):
    count=0
    url="https://graph.facebook.com/v2.10/"+id+"/likes?access_token="+token
    while(True):
        r = requests.get(url)
        r=r.json()
        #print(r)
        #for i in range(len(r['data'])):
            #names=r['data'][i]['name']
            #print(names)
        if('data' in r):
            count=count+len(r['data'])
        else:
            break
        if not 'paging' in r:
            break
        else:
            if not 'next' in r['paging']:
                break
        url=r['paging']['next']
    return count
        
def get_comments_on_post(id):
    count=0
    len_of_comments=0;
    url="https://graph.facebook.com/v2.10/"+id+"/comments?access_token="+token
    while(True):
        r = requests.get(url)
        r=r.json()
        break
    if('data' in r):
        count=count+len(r['data'])
        for comment in r['data']:
            url="https://graph.facebook.com/v2.10/"+comment['id']+"?fields=comment_count&access_token="+token
            len_of_comments=len_of_comments+len(comment['message'])
            r=requests.get(url).json()
            count=count+r['comment_count']
    return count,len_of_comments
        
    

#token = "EAACEdEose0cBAHZBIMseOEEIwTGcI29xByWpfFtOQohBvJj00ZA2ZCSdZAD7v6BtjfLthkGs5JjmBUuwXuPwtv533kMNfGzYFErxrVnBrJIEUwkIwO3CZB53L0wAneajbSu2HtKJD2pPVDO1pt3Y3cFU8yJBMS9PZA7MdDHZBLx1wZBgT0wJa22ZB3md3vmxxIXxZCKaKEPnUUiAZDZD"

facebook=fb.graph.api(token)
url = "https://graph.facebook.com/v2.10/me?fields=feed,name,email,friends,education,birthday,age_range,gender&access_token="+token    
object=requests.get(url).json()
facebook.show_fields(object)

csvfile="fb_data_sheet_4.csv"

#name
#tags
#links
#dateofbirth
#school year
#post
#gender
#Likes_count
#friends_count

friends_id=[]
fri = object['friends']
for friend in fri['data']:
    friends_id.append(friend['id'])
    print(friend['name'],friend['id'])

csv_file=open(csvfile,"w")
wr=csv.writer(csv_file)

for fid in friends_id:
    #print(friend['name']+"-"+friend['id'])    
    url = "https://graph.facebook.com/v2.10/"+fid+"?fields=feed,name,email,friends,education,birthday,age_range,gender&access_token="+token    
    object=requests.get(url).json()
    educ_year="NULL"
    #print(object['name'])
    name=object['name']
    if('education' in object):
        #print(object['education'][0]['school']['name'])
        for i in object['education']:
            if('year' in i):
            #print(object['education'][0]['year'])
                if('name' in i['year']):
                    educ_year=i['year']['name']
                else:
                    break
            else:
                break
    else:
       educ_year="NULL"
    if('birthday' in object):
        b_year=object['birthday']
    else:
        b_year="NULL"
    st=[]
    msg=[]
    did=[]
    if('feed' in object ):
        for feeds in object['feed']['data']:
            if('story' in feeds):
                #print(feeds['story'])
                st.append(feeds['story'])
            else:
                st.append("NULL")
            if('message' in feeds):
                #print(feeds['message'])
                msg.append(feeds['message'])
            else:
                msg.append("NULL")
            if('id' in feeds):
                did.append(feeds['id'])
            else:
                did.append("NULL")             
                
    if('age_range' in object):
        #print(object['age_range'])
        agerange=object['age_range']
    else:
        agerange="NULL"
    if('gender' in object):
        gender=object['gender']
    else:
        gender="NULL"
    if 'friends' in object:
        number_of_friends=object['friends']['summary']['total_count']
    else:
        number_of_friends=500
    l=len(msg)

    graph = fbsdk.GraphAPI(token)
    likes=get_myfacebook_likes(graph,fid)
    number_of_likes=len(likes)
    
    
    for i in range(l):
        line=str(name)+","+str(fid)+","+st[i]+","+msg[i]+","+str(gender)+","+str(number_of_friends)+"\n"
        #print(name)
        #print(st[i])
        #print(msg[i])
        #print(number_of_friends)
        story=st[i]
        #story=story.encode('cp1252').decode('utf-8')
        story=unicode(story).encode("utf-8")
        #story.decode('utf-8')
        message=msg[i]
        #message=message.encode('cp1252').decode('utf-8')
        message=unicode(message).encode("utf-8")
        #message.decode('utf-8')
        #print(message)
        #print(story)
       
        Total_post_likes=0
        Total_comments=0
        Total_len_of_comments=0
        if(str(did[i])!="NULL" and st[i]!="NULL"):
            number_of_likes_on_post=get_likes_on_post(did[i])
            number_of_reactions=get_reactions_on_post(did[i])
            number_of_comments, Total_len_of_comments=get_comments_on_post(did[i])
           
            Total_post_likes=number_of_likes_on_post+ number_of_reactions
            Total_comments=number_of_comments
        row=[]
        row.append(name)    
        row.append(fid)
        row.append(str(b_year))
        row.append(str(educ_year))
        row.append(story)
        row.append(message)
        row.append(len(message))
        row.append(Total_post_likes)
        row.append(Total_comments)
        row.append(Total_len_of_comments)
        row.append(gender)
        row.append(number_of_friends)
        row.append(number_of_likes)
        row.append("Teenager")
        #wr = csv.writer(csv_file)
        print(row)
        wr.writerow(row)
        #print(row[4]+','+row[5])
        del row
