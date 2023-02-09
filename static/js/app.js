// document.addEventListener('DOMContentLoaded', () => {
//
//     const socket = io();
//     let s1 = 'X'
//     let s2 = 'O'
//     let c1 = 'green'
//     let c2 = 'yellow'
//     let gamer = true;
//     let offline = true;
//     let xScore = 0;
//     let oScore = 0;
//     let symbol = s1;
//     let boardSize = 20;
//     let moveCount;
//     let ajdee = 0;
//     let room = null
//     socket.on('id',(data)=>{
//         if(ajdee===0){
//             ajdee=data.id;
//             gamer=data.turn;
//             offline = false;
//             room = data.room
//         }
//     });
//     const msg = $("#msg");
//     const chatForm = $("#chat-form");
//     const chatInput = $("#chat-input");
//
//     chatForm.on('submit', (e)=>{
//         e.preventDefault();
//         socket.emit('msg recieved',{message:chatInput.val(),room});
//         chatInput.val('');
//     });
//
//     socket.on('chat msg',(message)=>{
//         msg.prepend(`
//         <div class="singleMessage">
//             <div class="span" style="color:rgb(85, 36, 126)">${message.playerName}:</div>
//             <div class="span" style="inline-block">${message.messageVal}</div>
//         </div>
//         `)
//     })
//     //fills the board with boxes with coordinates
//     function makeNewBoard() {
//         $('#board').empty()
//         for (let i = 0; i < boardSize; i++) {
//             for (let j = 0; j < boardSize; j++) {
//                 let square = `<div class='square' data-xy=${i+','+j}></div>`;
//                 $('#board').append(square);
//             }
//         }
//         $('#board').off();
//         moveCount = 1;
//         addClicks()
//         addHover()
//         $('.winner__alert').removeClass('winner').text('')
//     }
//
//
//
//     //on kilck places a symbol of one of the players
//     function addClicks() {
//         $('.square').on('click', function (e) {
//             e.preventDefault();
//             if ($(this).text() === '' && (gamer === true || offline)) {
//                 $(this).toggleClass('rotator');
//                 if(!offline){
//                     gamer = false;
//                     symbol = s1;
//                 }else{
//                     gamer = gamer === true ? false : true;
//                     symbol = gamer === true ? s1 : s2;
//                 }
//                 $(this).text(symbol);
//                 let color = symbol===s1?c1:c2;
//                 $(this).css('background',color);
//                 moveCount++;
//                 let coords = $(this).data("xy");
//                 //socket.emit('board update',{coords, id:ajdee, room});
//             }
//         });
//     }
//     function addHover(){
//         $('.square').mouseenter(function(){
//             if($(this).text()===''){
//                 if(!gamer && offline)       $(this).css('background-color',c1);
//                 else if(gamer && offline) $(this).css('background-color',c2);
//                 else if(!offline)          $(this).css('background-color',c1);}
//             let coords = $(this).data("xy");
//             socket.emit('hoverin',{coords,room,id:ajdee})
//         })
//         $('.square').mouseleave(function(){
//             if($(this).text()===''){  $(this).css('background-color','rgb(235, 235, 207)');}
//             let coords = $(this).data("xy");
//             socket.emit('hoverout',{coords,room,id:ajdee})
//         })
//     }
//
//     socket.on('server update',(data)=>{
//         const { room, coords, id, turn } = data
//         moveCount++;
//         checkForWinner()
//         if( ajdee !== id ){
//             $(`div[data-xy = "${ coords }"]`).text(s2).css('background',c2).addClass('incomin');
//             gamer = true;
//         }
//     });
//     socket.on('make new board', ()=>{makeNewBoard()})
//
//     socket.on('server hoverin',(data)=>{
//         const { room, coords, id} = data
//         if(ajdee!==id){
//             if($(`div[data-xy = "${ coords }"]`).text()==s1){
//                 $(`div[data-xy = "${ coords }"]`).css('transform','scale(.9)');
//             }else if($(`div[data-xy = "${ coords }"]`).text()===s2){
//                 $(`div[data-xy = "${ coords }"]`).removeClass('incomin').css('transform','scale(.9)');
//             }else{
//                 $(`div[data-xy = "${ coords }"]`).css('background',c2).css('transform','scale(.9)');
//             }
//         }
//     });
//
//     socket.on('server hoverout', (data)=>{
//         const { room, coords ,id } = data
//         if(ajdee!==id){
//             if($(`div[data-xy = "${ coords }"]`).text()===s1){
//                 $(`div[data-xy = "${ coords }"]`).css('transform','scale(1)');
//             }else if($(`div[data-xy = "${ coords }"]`).text()===s2){
//                 $(`div[data-xy = "${ coords }"]`).css('transform','scale(1)');
//             }else{
//                 $(`div[data-xy = "${ coords }"]`).css('background','rgb(235, 235, 207)').css('transform','scale(1)');
//             }
//         }
//     })
//
//
//     $('.winner__alert').on('click',()=>{
//         if(!offline){ socket.emit( 'make new board')  } else { makeNewBoard() };
//     })
//
//     function winner(score){
//         if(score==='X'){
//             socket.emit('winner', { id:ajdee, room})
//         }
//     }
//     socket.on('server winner', data => {
//         const { id } = data
//         let c
//         if(id===ajdee){
//             xScore++
//             $('.xScore').text(xScore)
//             c = s1
//         } else {
//             oScore++
//             $('.oScore').text(oScore)
//             c=s2
//         }
//         $('.winner__alert')
//             .addClass('winner')
//             .html(`<p>the winner is </p><spam>${c}</spam><p> click here to play again</p>`);
//     })
//
//     function checkForWinner() {
//         let $square = $('.square');
//         for (let i = 0; i < boardSize * boardSize; i++) {
//             //checking for winner in horizontal plane
//             if ($square.eq(i).text() !== '' &&
//                 $square.eq(i).text() === $square.eq(i + 1).text() &&
//                 $square.eq(i + 1).text() === $square.eq(i + 2).text() &&
//                 $square.eq(i + 2).text() === $square.eq(i + 3).text() &&
//                 $square.eq(i + 3).text() === $square.eq(i + 4).text() &&
//                 i%boardSize<16) {
//                     $square.eq(i).css("background-color","red");
//                     $square.eq(i+1).css("background-color","red");
//                     $square.eq(i+2).css("background-color","red");
//                     $square.eq(i+3).css("background-color","red");
//                     $square.eq(i+4).css("background-color","red");
//                 winner($square.eq(i).text());
//             }
//             //checking for winner in vertical plane
//             else if ($square.eq(i).text() !== '' &&
//                 $square.eq(i).text() === $square.eq(i + 20).text() &&
//                 $square.eq(i + 20).text() === $square.eq(i + 40).text() &&
//                 $square.eq(i + 40).text() === $square.eq(i + 60).text() &&
//                 $square.eq(i + 60).text() === $square.eq(i + 80).text()) {
//                     $square.eq(i).css("background-color","red");
//                     $square.eq(i+20).css("background-color","red");
//                     $square.eq(i+40).css("background-color","red");
//                     $square.eq(i+60).css("background-color","red");
//                     $square.eq(i+80).css("background-color","red");
//                 winner($square.eq(i).text());
//             }
//             //checking for winner on a diagonal forward plane
//             else if ($square.eq(i).text() !== '' &&
//                 $square.eq(i).text() === $square.eq(i + 21).text() &&
//                 $square.eq(i + 21).text() === $square.eq(i + 42).text() &&
//                 $square.eq(i + 42).text() === $square.eq(i + 63).text() &&
//                 $square.eq(i + 63).text() === $square.eq(i + 84).text() &&
//                 i%boardSize<16) {
//                     $square.eq(i).css("background-color","red");
//                     $square.eq(i+21).css("background-color","red");
//                     $square.eq(i+42).css("background-color","red");
//                     $square.eq(i+63).css("background-color","red");
//                     $square.eq(i+84).css("background-color","red");
//                 winner($square.eq(i).text());
//             }
//             //checking for winner in a minus diagonal option
//             else if ($square.eq(i).text() !== '' &&
//                 $square.eq(i).text() === $square.eq(i + 19).text() &&
//                 $square.eq(i + 19).text() === $square.eq(i + 38).text() &&
//                 $square.eq(i + 38).text() === $square.eq(i + 57).text() &&
//                 $square.eq(i + 57).text() === $square.eq(i + 76).text() &&
//                 i%boardSize>3
//                 ) {
//                     $square.eq(i).css("background-color","red");
//                     $square.eq(i+19).css("background-color","red");
//                     $square.eq(i+38).css("background-color","red");
//                     $square.eq(i+57).css("background-color","red");
//                     $square.eq(i+76).css("background-color","red");
//                 winner($square.eq(i).text());
//             }
//             //checking if board is completely full
//             else if (moveCount === 400){
//                 makeNewBoard();
//             }
//         }
//     }
//
//     makeNewBoard();
//
// })