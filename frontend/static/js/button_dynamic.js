function add_text(name_parameter){
    var new_box = document.createElement('p');
    new_box.innerHTML = "<input id=".concat(name_parameter, "_text",  " type='text'>");
    var new_button = document.createElement('button');
    new_button.innerHTML = '확인';
    new_box.appendChild(new_button);
    document.getElementById(name_parameter).appendChild(new_box);
}
/*
function append_button(id_parameter, onclick_parameter){
    var newButton = document.createElement('button');
    newButton.id = id_parameter;
    newButton.onclick = "return add_text('student', 'student_id')";
    //document.getElementById()
}
function save_database_attribute(text_element_id, element){
    var text_value = text_element_id.concat("_text");
    var content_value = document.getElementById(text_value).value;
    localStorage.setItem(element, content_value);
}

function appearstudent(){
    document.getElementById('age').addEventListener('click', function() {
        var newButton = document.createElement('button');
        newButton.innerHTML = '학번';
        document.getElementById('student').appendChild(newButton);
    });
}
function appearmajor(){
    document.getElementById('student').addEventListener('click', function() {
        var newButton = document.createElement('button');
        newButton.innerHTML = '전공';
        document.getElementById('major').appendChild(newButton);
    });
}
function appearbedtime(){
    document.getElementById('major').addEventListener('click', function() {
        var newButton = document.createElement('button');
        newButton.innerHTML = '취침시간';
        document.getElementById('bed_time').appendChild(newButton);
    });
}
function appearcleanrate(){
    document.getElementById('bed_time').addEventListener('click', function() {
        var newButton = document.createElement('button');
        newButton.innerHTML = '흡연여부';
        document.getElementById('cigaratte').appendChild(newButton);
    });
}

function appearsmoke(){
    document.getElementById('cigaratte').addEventListener('click', function() {
        var newButton = document.createElement('button');
        newButton.innerHTML = '청소주기';
        document.getElementById('clean_rate').appendChild(newButton);
    });
}

function appearcleanrate(){
    document.getElementById('clean_rate').addEventListener('click', function() {
        var newButton = document.createElement('button');
        newButton.innerHTML = '음주빈도';
        document.getElementById('drunk_rate').appendChild(newButton);
    });
}

function appeardrunk(){
    document.getElementById('drunk_rate').addEventListener('click', function() {
        var newButton = document.createElement('button');
        newButton.innerHTML = 'MBTI';
        document.getElementById('mbti').appendChild(newButton);
    });
}

function appearsmoke(){
    document.getElementById('major').addEventListener('click', function() {
        var newButton = document.createElement('button');
        newButton.innerHTML = '취침시간';
        document.getElementById('bed_time').appendChild(newButton);
    });
}
*/