-- scriptTest.lua (in your scripts directory)
local M = {}
 
local function testFunction()
      print("Test function called")
end
M.testFunction = testFunction
 
return M
