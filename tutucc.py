import argparse
import sys
from enum import Enum

class ErrorCode(Enum):
    UNEXPECTED_TOKEN = 'Unexpected token'
    ID_NOT_FOUND     = 'Identifier not found'
    DUPLICATE_ID     = 'Duplicate id found'


class Error(Exception):
    def __init__(self, error_code=None, token=None, message=None):
        self.error_code = error_code
        self.token = token
        # add exception class name before the message
        self.message = f'{self.__class__.__name__}: {message}'


class LexerError(Error):
    pass

class ParserError(Error):
    pass

###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################

class TokenType(Enum):
    # single-character token types
    PLUS          = '+'
    MINUS         = '-'
    MUL           = '*'
    DIV           = '/'
    LPAREN        = '('
    RPAREN        = ')'
    SEMI          = ';'
    DOT           = '.'
    COLON         = ':'
    COMMA         = ','
    LT            = '<'
    LE            = '<='
    GT            = '>'
    GE            = '>='
    NE            = '!='
    EQ            = '=='
    # block of reserved words
    INT           = 'INT'
    RETURN        = 'RETURN'
    # misc
    ID            = 'ID'
    INTEGER_CONST = 'INTEGER_CONST'
    REAL_CONST    = 'REAL_CONST'
    ASSIGN        = '='
    EOF           = 'EOF'
    EXPRSTMT      = 'EXPRSTMT'
    VAR           = 'VAR'


class Token:
    def __init__(self, type, value, lineno=None, column=None):
        self.type = type
        self.value = value
        self.lineno = lineno
        self.column = column

    def __str__(self):
        """String representation of the class instance.

        Example:
            >>> Token(TokenType.INTEGER, 7, lineno=5, column=10)
            Token(TokenType.INTEGER, 7, position=5:10)
        """
        return 'Token({type}, {value}, position={lineno}:{column})'.format(
            type=self.type,
            value=repr(self.value),
            lineno=self.lineno,
            column=self.column,
        )

    def __repr__(self):
        return self.__str__()


def _build_reserved_keywords():
    """Build a dictionary of reserved keywords.

    The function relies on the fact that in the TokenType
    enumeration the beginning of the block of reserved keywords is
    marked with PROGRAM and the end of the block is marked with
    the END keyword.

    Result:
        {'PROGRAM': <TokenType.PROGRAM: 'PROGRAM'>,
         'INTEGER': <TokenType.INTEGER: 'INTEGER'>,
         'REAL': <TokenType.REAL: 'REAL'>,
         'DIV': <TokenType.INTEGER_DIV: 'DIV'>,
         'VAR': <TokenType.VAR: 'VAR'>,
         'PROCEDURE': <TokenType.PROCEDURE: 'PROCEDURE'>,
         'BEGIN': <TokenType.BEGIN: 'BEGIN'>,
         'END': <TokenType.END: 'END'>}
    """
    # enumerations support iteration, in definition order
    tt_list = list(TokenType)
    start_index = tt_list.index(TokenType.INT)
    end_index = tt_list.index(TokenType.RETURN)
    reserved_keywords = {
        token_type.value: token_type
        for token_type in tt_list[start_index:end_index + 1]
    }
    return reserved_keywords


RESERVED_KEYWORDS = _build_reserved_keywords()


class Lexer:
    def __init__(self, text):
        # client string input, e.g. "4 + 2 * 3 - 6 / 2"
        self.text = text
        # self.pos is an index into self.text
        self.pos = 0
        self.current_char = self.text[self.pos]
        # token line number and column number
        self.lineno = 1
        self.column = 1

    def error(self):
        s = "Lexer error on '{lexeme}' line: {lineno} column: {column}".format(
            lexeme=self.current_char,
            lineno=self.lineno,
            column=self.column,
        )
        raise LexerError(message=s)

    def advance(self):
        """Advance the `pos` pointer and set the `current_char` variable."""
        if self.current_char == '\n':
            self.lineno += 1
            self.column = 0

        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None  # Indicates end of input
        else:
            self.current_char = self.text[self.pos]
            self.column += 1

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos > len(self.text) - 1:
            return None
        else:
            return self.text[peek_pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def number(self):
        """Return a (multidigit) integer or float consumed from the input."""

        # Create a new token with current line and column number
        token = Token(type=None, value=None, lineno=self.lineno, column=self.column)

        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        if self.current_char == '.':
            result += self.current_char
            self.advance()

            while self.current_char is not None and self.current_char.isdigit():
                result += self.current_char
                self.advance()

            token.type = TokenType.REAL_CONST
            token.value = float(result)
        else:
            token.type = TokenType.INTEGER_CONST
            token.value = int(result)

        return token

    def _id(self):
        """Handle identifiers and reserved keywords"""

        # Create a new token with current line and column number
        token = Token(type=None, value=None, lineno=self.lineno, column=self.column)

        value = ''
        while self.current_char is not None and self.current_char.isalnum():
            value += self.current_char
            self.advance()

        token_type = RESERVED_KEYWORDS.get(value.upper())
        if token_type is None:
            token.type = TokenType.ID
            token.value = value
        else:
            # reserved keyword
            token.type = token_type
            token.value = value.upper()

        return token

    def get_next_token(self):
        """Lexical analyzer (also known as scanner or tokenizer)

        This method is responsible for breaking a sentence
        apart into tokens. One token at a time.
        """
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isalpha():
                return self._id()

            if self.current_char.isdigit():
                return self.number()

            if self.current_char == '>' and self.peek() == '=':
                token = Token(
                    type=TokenType.GE,
                    value='>=',
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                self.advance()
                return token

            if self.current_char == '>':
                token = Token(
                    type=TokenType.GT,
                    value='>',
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                return token

            if self.current_char == '<' and self.peek() == '=':
                token = Token(
                    type=TokenType.LE,
                    value='<=',
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                self.advance()
                return token

            if self.current_char == '<':
                token = Token(
                    type=TokenType.LT,
                    value='<',
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                return token

            if self.current_char == '=' and self.peek() == '=':
                token = Token(
                    type=TokenType.EQ,
                    value='==',
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                self.advance()
                return token

            if self.current_char == '=':
                token = Token(
                    type=TokenType.ASSIGN,
                    value='=',
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                return token

            if self.current_char == '!' and self.peek() == '=':
                token = Token(
                    type=TokenType.NE,
                    value=TokenType.NE.value,  # ':='
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                self.advance()
                return token

            # single-character token
            try:
                # get enum member by value, e.g.
                # TokenType(';') --> TokenType.SEMI
                token_type = TokenType(self.current_char)
            except ValueError:
                # no enum member with value equal to self.current_char
                self.error()
            else:
                # create a token with a single-character lexeme as its value
                token = Token(
                    type=token_type,
                    value=token_type.value,  # e.g. ';', '.', etc
                    lineno=self.lineno,
                    column=self.column,
                )
                self.advance()
                return token

        # EOF (end-of-file) token indicates that there is no more
        # input left for lexical analysis
        return Token(type=TokenType.EOF, value=None)

class AST:
    pass

class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right

class UnaryOp(AST):
    def __init__(self, op, expr):
        self.token = self.op = op
        self.expr = expr

class Assign(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right

class Num(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value

class ExprStmt(AST):
    def __init__(self, expr):
        self.token = Token(TokenType.EXPRSTMT, None)
        self.expr = expr

class Var(AST):
    def __init__(self, name, offset):
        self.name = name
        self.offset = offset
        self.token = Token(TokenType.VAR, None)

class Function(AST):
    def __init__(self, nodes, locals, stack_size):
        self.nodes = nodes
        self.locals = locals
        self.stack_size = stack_size

class Parser:
    """
    program    = stmt*
    stmt       = expr ";"
    expr       = assign
    assign     = equality ("=" assign)?
    equality   = relational ("==" relational | "!=" relational)*
    relational = add ("<" add | "<=" add | ">" add | ">=" add)*
    add        = mul ("+" mul | "-" mul)*
    mul        = unary ("*" unary | "/" unary)*
    unary      = ("+" | "-")? primary
    primary    = num | ident | "(" expr ")"
    """
    def __init__(self, lexer):
        self.lexer = lexer
        # set current token to the first token taken from the input
        self.current_token = self.get_next_token()
        self.locals = []

    def find_var(self, token):
        for var in self.locals:
            if var.name == token.value:
                return var
        return None

    def get_next_token(self):
        return self.lexer.get_next_token()

    def error(self, error_code, token):
        raise ParserError(
            error_code=error_code,
            token=token,
            message=f'{error_code.value} -> {token}',
        )

    def eat(self, token_type):
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.get_next_token()
        else:
            self.error(
                error_code=ErrorCode.UNEXPECTED_TOKEN,
                token=self.current_token,
            )

    def program(self):
        self.locals = []
        stmts = []

        while self.current_token.type != TokenType.EOF:
            stmts.append(self.stmt())

        prog = Function(stmts, self.locals, 0)
        return prog

    def stmt(self):
        token = self.current_token
        if token.type == TokenType.RETURN:
            self.eat(TokenType.RETURN)
            node = UnaryOp(op=token, expr=self.expr())
            self.eat(TokenType.SEMI)
            return node

        node = ExprStmt(self.expr())
        if self.current_token.type == TokenType.SEMI:
            self.eat(TokenType.SEMI)
        return node

    def expr(self):
        return self.assign()

    def assign(self):
        node = self.equality()

        token = self.current_token
        if token.type == TokenType.ASSIGN:
            self.eat(TokenType.ASSIGN)
            node = Assign(left=node, op=token, right=self.assign())
        return node

    def equality(self):
        node = self.relational()

        while self.current_token.type in (TokenType.EQ, TokenType.NE):
            token = self.current_token
            if token.type == TokenType.EQ:
                self.eat(TokenType.EQ)
            elif token.type == TokenType.NE:
                self.eat(TokenType.NE)
            
            node = BinOp(left=node, op=token, right=self.relational())

        return node

    def relational(self):
        node = self.add()

        while self.current_token.type in (TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE):
            token = self.current_token
            if token.type == TokenType.LT:
                self.eat(TokenType.LT)
            elif token.type == TokenType.LE:
                self.eat(TokenType.LE)
            elif token.type == TokenType.GT:
                self.eat(TokenType.GT)
            elif token.type == TokenType.GE:
                self.eat(TokenType.GE)

            node = BinOp(left=node, op=token, right=self.add())

        return node

    def add(self):
        node = self.mul()

        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            token = self.current_token
            if token.type == TokenType.PLUS:
                self.eat(TokenType.PLUS)
            elif token.type == TokenType.MINUS:
                self.eat(TokenType.MINUS)
            
            node = BinOp(left=node, op=token, right=self.mul())

        return node

    def mul(self):
        node = self.unary()

        while self.current_token.type in (TokenType.MUL, TokenType.DIV):
            token = self.current_token
            if token.type == TokenType.MUL:
                self.eat(TokenType.MUL)
            elif token.type == TokenType.DIV:
                self.eat(TokenType.DIV)

            node = BinOp(left=node, op=token, right=self.unary())

        return node

    def primary(self):
        token = self.current_token
        if token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.expr()
            self.eat(TokenType.RPAREN)
            return node

        if token.type == TokenType.ID:
            self.eat(TokenType.ID)
            var = self.find_var(token)
            if var is None:
                var = Var(token.value, 0)
                self.locals.append(var)
            return var


        self.current_token = self.get_next_token()

        return Num(token)

    def unary(self):
        token = self.current_token
        if token.type == TokenType.PLUS:
            self.eat(TokenType.PLUS)
            return self.primary()
        if token.type == TokenType.MINUS:
            self.eat(TokenType.MINUS)
            zero = Num(Token(type=TokenType.INTEGER_CONST, value=0))
            return BinOp(left=zero, op=token, right=self.unary())
        return self.primary()

    def gen_addr(self, node):
        if node.token.type == TokenType.VAR:
            print("  lea rax, [rbp-%d]" % node.offset)
            print("  push rax")
            return
        
        self.error(
            error_code=ErrorCode.UNEXPECTED_TOKEN,
            token=self.current_token,
        )

    def load(self):
        print("  pop rax")
        print("  mov rax, [rax]")
        print("  push rax")

    def store(self):
        print("  pop rdi")
        print("  pop rax")
        print("  mov [rax], rdi")
        print("  push rdi")

    def code_gen(self, node):
        if node.token.type == TokenType.INTEGER_CONST:
            print("  push %s" % node.value)
            return
        if node.token.type == TokenType.EXPRSTMT:
            self.code_gen(node.expr)
            print("  add rsp, 8")
            return
        if node.token.type == TokenType.VAR:
            self.gen_addr(node)
            self.load()
            return
        if node.token.type == TokenType.ASSIGN:
            self.gen_addr(node.left)
            self.code_gen(node.right)
            self.store()
            return
        if node.token.type == TokenType.RETURN:
            self.code_gen(node.expr)
            print("  pop rax")
            print("  jmp .L.return")
            return

        self.code_gen(node.left)
        self.code_gen(node.right)

        print("  pop rdi")
        print("  pop rax")

        if node.op.type == TokenType.PLUS:
            print("  add rax, rdi")
        if node.op.type == TokenType.MINUS:
            print("  sub rax, rdi")
        if node.op.type == TokenType.MUL:
            print("  imul rax, rdi")
        if node.op.type == TokenType.DIV:
            print("  cqo")
            print("  idiv rdi")
        if node.op.type == TokenType.EQ:
            print("  cmp rax, rdi")
            print("  sete al")
            print("  movzb rax, al")
        if node.op.type == TokenType.NE:
            print("  cmp rax, rdi")
            print("  setne al")
            print("  movzb rax, al")
        if node.op.type == TokenType.LT:
            print("  cmp rax, rdi")
            print("  setl al")
            print("  movzb rax, al")
        if node.op.type == TokenType.LE:
            print("  cmp rax, rdi")
            print("  setle al")
            print("  movzb rax, al")
        if node.op.type == TokenType.GT:
            print("  cmp rax, rdi")
            print("  setg al")
            print("  movzb rdi, al")
        if node.op.type == TokenType.GE:
            print("  cmp rax, rdi")
            print("  setge al")
            print("  movzb rdi, al")

        print("  push rax")

    def parse(self):
        nodes = self.program()
        if self.current_token.type != TokenType.EOF:
            self.error(
                error_code=ErrorCode.UNEXPECTED_TOKEN,
                token=self.current_token,
            )

        return nodes


def main():
    parser = argparse.ArgumentParser(
        description='tutucc - simple c compiler'
    )
    parser.add_argument('inputfile', help='c source file')
    args = parser.parse_args()

    text = args.inputfile

    lexer = Lexer(text)
    try:
        parser = Parser(lexer)
        prog = parser.parse()
        offset = 0
        for v in prog.locals:
            offset = offset + 8
            v.offset = offset
        prog.stack_size = offset
        print(".intel_syntax noprefix")
        print(".global main")
        print("main:")
        print("  push rbp")
        print("  mov rbp, rsp")
        print("  sub rsp, %d" % prog.stack_size)
        for n in prog.nodes:
            parser.code_gen(n)
        print(".L.return:")
        print("  mov rsp, rbp")
        print("  pop rbp")
        print("  ret")
        sys.exit(0)
    except (LexerError, ParserError) as e:
        print(e.message)
        sys.exit(1)

if __name__ == '__main__':
    main()