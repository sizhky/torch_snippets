from IPython.display import display, HTML
from .loader import rand
def displayTextForCopy(string):
    elemId = 'X'+rand()+rand()
    funcName = 'Y'+rand()+rand()
    display(HTML('''
        <script>
        function myFunction() {
          /* Get the text field */
          var copyText = document.getElementById("myInput");

          /* Select the text field */
          copyText.select();
          copyText.setSelectionRange(0, 99999); /*For mobile devices*/

          /* Copy the text inside the text field */
          document.execCommand("copy");

        }

        </script>
        <!-- The text field -->
        <input type="text" value="mystring" id="myInput">

        <!-- The button used to copy the text -->
        <button onclick="myFunction()">Copy text</button>
        '''.replace('mystring',string).replace('myInput',elemId).replace('myFunction', funcName)
    ))
