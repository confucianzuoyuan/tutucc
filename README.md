# tutucc

`tutucc.py` is a Compiler For C Language Written in Python.

`tutucc.py`是使用`Python`实现的C语言编译器。

**EBNF**

```
program = (global-var | function)*

basetype = builtin-type | struct-decl | typedef-name | enum-specifier

builtin-type = "void" | "_Bool" | "char" | "short" | "int"
             | "long" | "long" "long"

declarator = "*"* ("(" declarator ")" | ident) type-suffix

abstract-declarator = "*"* ("(" abstract-declarator ")")? type-suffix

type-suffix = ("[" num "]" type-suffix)?

type-name = basetype abstract-declarator type-suffix

struct-decl = "struct" ident
            | "struct" ident? "{" struct-member "}"

enum-specifier = "enum" ident
               | "enum" ident? "{" enum-list? "}"

enum-list = ident ("=" num)? ("," ident ("=" num)?)* ","?

struct-member = basetype declarator type-suffix ";"

function = basetype declarator "(" params? ")" ("{" stmt* "}" | ";")
params   = param ("," param)*
param    = basetype declarator type-suffix

global-var = basetype declarator type-suffix ";"

declaration = basetype declarator type-suffix ("=" expr)? ";"
            | basetype ";"

stmt2 = "return" expr ";"
      | "if" "(" expr ")" stmt ("else" stmt)?
      | "while" "(" expr ")" stmt
      | "for" "(" (expr? ";" | declaration) expr? ";" expr? ")" stmt
      | "{" stmt* "}"
      | declaration
      | expr ";"

expr = assign ("," assign)*

assign    = equality (assign-op assign)?
assign-op = "=" | "+=" | "-=" | "*=" | "/="

equality = relational ("==" relational | "!=" relational)*
relational = add ("<" add | "<=" add | ">" add | ">=" add)*

add = mul ("+" mul | "-" mul)*
mul = cast ("*" cast | "/" cast)*

cast = "(" type-name ")" cast | unary

unary = ("+" | "-" | "*" | "&")? cast
      | ("++" | "--") unary
      | postfix

postfix = primary ("[" expr "]" | "." ident | "->" ident | "++" | "--")*

stmt-expr = "(" "{" stmt stmt* "}" ")"

func-args = "(" (assign ("," assign)*)? ")"

primary = "(" "{" stmt-expr-tail
        | "(" expr ")"
        | "sizeof" "(" type-name ")"
        | "sizeof" unary
        | ident func-args?
        | str
        | num

```